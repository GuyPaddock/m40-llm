use m40_llm::decode::StoppingCriteria;

#[test]
fn stops_on_max_tokens() {
    let sc = StoppingCriteria::new(Some(3), None);
    assert!(!sc.should_stop(&[]));
    assert!(!sc.should_stop(&[1]));
    assert!(!sc.should_stop(&[1, 2]));
    assert!(sc.should_stop(&[1, 2, 3]));
}

#[test]
fn stops_on_eos() {
    let sc = StoppingCriteria::new(None, Some(42));
    assert!(!sc.should_stop(&[]));
    assert!(!sc.should_stop(&[1, 2]));
    assert!(sc.should_stop(&[1, 2, 42]));
}

#[test]
fn stops_on_either_condition() {
    let sc = StoppingCriteria::new(Some(5), Some(9));
    assert!(!sc.should_stop(&[1, 2, 3]));
    assert!(sc.should_stop(&[1, 2, 3, 4, 5]));
    assert!(sc.should_stop(&[1, 9]));
}
