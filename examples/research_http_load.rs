//! Retain individual HTTP arrival times for one or several requests to one M40 server.
use anyhow::{ensure, Context, Result};
use clap::Parser;
use reqwest::Client;
use serde_json::{json, Value};
use std::{
    fs,
    io::Write,
    path::PathBuf,
    time::{Duration, Instant, SystemTime, UNIX_EPOCH},
};
use tokio::task::JoinSet;

#[derive(Parser)]
struct Args {
    /// Base URL of the running normal /generate server.
    #[arg(long)]
    server: String,
    /// One raw JSON request per client, in submission order.
    #[arg(long, required = true, num_args=1..)]
    requests: Vec<PathBuf>,
    /// Delay each client from a common start, in milliseconds.
    #[arg(long, value_delimiter = ',')]
    offsets_ms: Vec<u64>,
    #[arg(long)]
    output: PathBuf,
}

fn millis(start: Instant) -> f64 {
    start.elapsed().as_secs_f64() * 1000.0
}

async fn send_request(
    client: Client,
    url: String,
    body: Vec<u8>,
    streaming: bool,
    slot: usize,
    delay: u64,
    origin: Instant,
) -> Result<Value> {
    tokio::time::sleep_until((origin + Duration::from_millis(delay)).into()).await;
    let submitted_ms = millis(origin);
    let mut response = client
        .post(url)
        .header("Content-Type", "application/json")
        .body(body)
        .send()
        .await?;
    let headers_ms = millis(origin);
    let status = response.status();
    if !status.is_success() {
        let detail = response.text().await?;
        return Ok(
            json!({"slot":slot,"status":status.as_u16(),"error":detail,"submitted_ms":submitted_ms,
            "headers_ms":headers_ms,"done_ms":millis(origin)}),
        );
    }
    if !streaming {
        let bytes = response.bytes().await?;
        let done_ms = millis(origin);
        let payload: Value = serde_json::from_slice(&bytes)?;
        ensure!(payload["output"].is_string(), "missing buffered output");
        return Ok(
            json!({"slot":slot,"status":status.as_u16(),"submitted_ms":submitted_ms,
            "headers_ms":headers_ms,"done_ms":done_ms,"wall_ms":done_ms-submitted_ms,"response":payload}),
        );
    }
    let mut tail = Vec::new();
    let mut arrivals = Vec::new();
    let mut complete = None;
    while let Some(bytes) = response.chunk().await? {
        let arrived_ms = millis(origin);
        tail.extend_from_slice(&bytes);
        let mut consumed = 0;
        for (i, _) in tail.iter().enumerate().filter(|(_, byte)| **byte == b'\n') {
            ensure!(complete.is_none(), "stream data after completion");
            let record: Value = serde_json::from_slice(&tail[consumed..i])?;
            let text = record["output"].as_str().context("stream output missing")?;
            if record["done"]
                .as_bool()
                .context("stream done flag missing")?
            {
                complete = Some(text.to_owned());
            } else {
                arrivals.push(arrived_ms);
            }
            consumed = i + 1;
        }
        tail.drain(..consumed);
    }
    let done_ms = millis(origin);
    ensure!(tail.is_empty(), "incomplete NDJSON line");
    Ok(
        json!({"slot":slot,"status":status.as_u16(),"submitted_ms":submitted_ms,
        "headers_ms":headers_ms,"done_ms":done_ms,"wall_ms":done_ms-submitted_ms,
        "arrival_ms":arrivals,"output":complete.context("stream never completed")?}),
    )
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    ensure!(
        args.offsets_ms.len() == args.requests.len(),
        "one offset per request required"
    );
    let mut file = fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(args.output)?;
    let client = Client::builder()
        .no_proxy()
        .timeout(Duration::from_secs(3600))
        .build()?;
    let url = format!("{}/generate", args.server.trim_end_matches('/'));
    let mut payloads = Vec::new();
    for path in &args.requests {
        let body = fs::read(path)?;
        let request: Value = serde_json::from_slice(&body)?;
        let stream = request["stream"]
            .as_bool()
            .context("explicit stream flag required")?;
        payloads.push((body, stream));
    }
    let unix_ns = SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos();
    let origin = Instant::now();
    let mut tasks = JoinSet::new();
    for (slot, ((body, stream), delay)) in payloads.into_iter().zip(&args.offsets_ms).enumerate() {
        tasks.spawn(send_request(
            client.clone(),
            url.clone(),
            body,
            stream,
            slot,
            *delay,
            origin,
        ));
    }
    let mut records = Vec::new();
    let mut errors = Vec::new();
    while let Some(task) = tasks.join_next().await {
        match task {
            Ok(Ok(record)) => records.push(record),
            Ok(Err(error)) => errors.push(format!("{error:#}")),
            Err(error) => errors.push(error.to_string()),
        }
    }
    records.sort_by_key(|r| r["slot"].as_u64());
    let success = errors.is_empty() && records.iter().all(|r| r["status"] == 200);
    serde_json::to_writer(
        &mut file,
        &json!({"client_origin_unix_ns":unix_ns.to_string(),"requests":args.requests,
        "offsets_ms":args.offsets_ms,"records":records,"errors":errors,"success":success}),
    )?;
    writeln!(file)?;
    ensure!(
        success,
        "one or more client requests failed; see retained response"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio::{
        io::{AsyncReadExt, AsyncWriteExt},
        net::TcpListener,
    };

    /// Serve split bytes on a real socket, including two records in one write.
    async fn stream_response(parts: &[&'static [u8]]) -> Result<Value> {
        let listener = TcpListener::bind(("127.0.0.1", 0)).await?;
        let address = listener.local_addr()?;
        let chunks = parts.to_vec();
        let server = tokio::spawn(async move {
            let (mut socket, _) = listener.accept().await?;
            let mut request = [0; 4096];
            socket.read(&mut request).await?;
            socket
                .write_all(b"HTTP/1.1 200 OK\r\nConnection: close\r\n\r\n")
                .await?;
            for chunk in chunks {
                socket.write_all(chunk).await?;
                tokio::time::sleep(Duration::from_millis(2)).await;
            }
            socket.shutdown().await
        });
        let client = Client::builder().no_proxy().build()?;
        let result = send_request(
            client,
            format!("http://{address}/generate"),
            b"{}".to_vec(),
            true,
            0,
            0,
            Instant::now(),
        )
        .await;
        server.await??;
        result
    }

    /// Each complete nonterminal record gets one arrival. The terminal cumulative
    /// text is retained without counting it as another token-boundary record.
    #[tokio::test]
    async fn collects_split_and_coalesced_records() -> Result<()> {
        let result = stream_response(&[
            br#"{"output":"a","done":"#,
            b"false}\n{\"output\":\"ab\",\"done\":false}\n{\"output\":\"ab\",\"done\":true}\n",
        ])
        .await?;
        let times = result["arrival_ms"].as_array().unwrap();
        assert_eq!(times.len(), 2);
        assert!(times[0].as_f64().unwrap() <= times[1].as_f64().unwrap());
        assert_eq!(result["output"], "ab");
        Ok(())
    }

    /// EOF after a partial record and EOF after a nonterminal record both
    /// invalidate a run, even when the HTTP status was successful.
    #[tokio::test]
    async fn rejects_incomplete_streams() {
        for body in [
            b"{\"output\":".as_slice(),
            b"{\"output\":\"a\",\"done\":false}\n",
        ] {
            assert!(stream_response(&[body]).await.is_err());
        }
    }
}
