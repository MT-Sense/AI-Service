from app.ml.predict import predict_employee_feedback


def test_predict_positive_feedback():
    text = "รู้สึกดีกับ สวัสดิการของบริษัท ที่คุยกันเข้าใจง่าย"
    result = predict_employee_feedback(text)

    print(f"ข้อความ: '{text}'")
    print(f"ผลประเมินทัศนคติ: {result['sentiment_label']}")

    assert result["sentiment_label"] in ("pos", "neg", "neu")
    assert "confidence" in result


if __name__ == "__main__":
    test_predict_positive_feedback()