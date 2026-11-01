use anyhow::Result;
use m40_llm::evaluation::{
    corpus::{Corpus, Split},
    scoring::{Fact, ScoringRule},
};
use serde_json::json;

fn corpus() -> Result<Corpus> {
    Corpus::parse(include_str!("../data/research_quality/v1.json"))
}

/// The frozen data must be loadable with both independent splits, and the
/// loader must reject reused IDs, normalized duplicates, and missing coverage.
#[test]
fn test_corpus_validates_disjoint_inputs_and_family_coverage() -> Result<()> {
    let original = corpus()?;
    assert!(!original.split(Split::Calibration).tasks.is_empty());
    assert!(!original.split(Split::HeldOut).passages.is_empty());
    let mut changed = original.clone();
    changed.held_out.passages[0].id = changed.calibration.passages[0].id.clone();
    assert!(changed.validate().is_err());
    let mut changed = original.clone();
    changed.held_out.tasks[0].prompt = format!(
        "  {} \n",
        changed.calibration.tasks[0].prompt.to_uppercase()
    );
    assert!(changed.validate().is_err());
    let mut changed = original.clone();
    changed.held_out.tasks.pop();
    assert!(changed.validate().is_err());
    let mut changed = original;
    changed.calibration.passages.clear();
    assert!(changed.validate().is_err());
    Ok(())
}

/// Reserved benchmark material must not enter calibration through prompts,
/// passages, or scoring answers. This checks the loader's data behavior.
#[test]
fn test_corpus_rejects_browser_prompt_and_answer_leakage() -> Result<()> {
    let original = corpus()?;
    for text in [
        "Explain everything that happens between the time a user types a web address into their browser and the time the page is fully loaded and displayed on their screen in as much detail as possible",
        "When a user types a web address into their browser and the page is fully loaded and displayed on their screen, a series of complex and often automated processes occur.",
    ] {
        let mut changed=original.clone();
        changed.calibration.passages[0].text=text.into();
        assert!(changed.validate().is_err());
        let mut changed=original.clone();
        changed.calibration.tasks[0].scoring=ScoringRule::Exact {answers:vec![text.replace(' ',"\n")]};
        assert!(changed.validate().is_err());
    }
    Ok(())
}

/// Exact responses allow harmless whitespace/case changes but not substring
/// matches, wrong values, extra prose, or completely unrelated fluent text.
#[test]
fn test_exact_answers_score_observable_response() -> Result<()> {
    let rule = ScoringRule::Exact {
        answers: vec!["14".into()],
    };
    assert!(rule.score(" 14\n")?.passed);
    for output in [
        "314",
        "14 or 31",
        "the answer is 14",
        "sunlight on a quiet street",
        "",
    ] {
        assert!(!rule.score(output)?.passed);
    }
    Ok(())
}

/// JSON grading compares scalar values and keys independent of order and
/// optional code fences, while exposing partial fact credit and format failure.
#[test]
fn test_json_facts_detect_wrong_values_and_extra_fields() -> Result<()> {
    let rule = ScoringRule::JsonObject {
        expected: json!({"day":"Monday","hall":12}),
    };
    assert!(
        rule.score("```json\n{\"hall\":12,\"day\":\"monday\"}\n```")?
            .passed
    );
    let wrong = rule.score("{\"hall\":3,\"day\":\"Monday\"}")?;
    assert!(!wrong.passed);
    assert_eq!(wrong.score, 0.5);
    assert_eq!(wrong.missing, ["hall"]);
    let extra = rule.score("{\"hall\":12,\"day\":\"Monday\",\"old_hall\":3}")?;
    assert!(!extra.passed && !extra.format_valid);
    assert!(!rule.score("{\"hall\":\"12\",\"day\":\"Monday\"}")?.passed);
    assert!(ScoringRule::JsonObject {
        expected: json!([])
    }
    .score("{}")
    .is_err());
    Ok(())
}

/// Concept coverage requires whole-word aliases and explicit structural
/// bounds; nonempty prose or an alias embedded in a different word cannot pass.
#[test]
fn test_concept_rubric_checks_coverage_and_structure() -> Result<()> {
    let rule = ScoringRule::Concepts {
        facts: vec![
            Fact {
                id: "scan".into(),
                aliases: vec!["scan".into(), "scanning".into()],
            },
            Fact {
                id: "backup".into(),
                aliases: vec!["backup".into()],
            },
        ],
        min_words: 6,
        min_paragraphs: 2,
    };
    let good =
        "Start scanning the photographs carefully.\r\n\r\nKeep a separate backup for recovery.";
    assert!(rule.score(good)?.passed);
    assert!(
        !rule
            .score("A scandal caused delays.\n\nA backup was kept for recovery.")?
            .passed
    );
    assert!(
        !rule
            .score("Scanning and backup in one short paragraph.")?
            .passed
    );
    let first = rule.score(good)?;
    assert_eq!(first, rule.score(good)?);
    Ok(())
}
