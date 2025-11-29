import "./Results.css";

const Results = ({ results }) => {
  if (!results) return null;

  return (
    <div className="results-wrapper">
      {results.predictions.map((pred, index) => (
        <div
          key={index}
          className={`result-box ${pred.prediction.toLowerCase()}`}
        >
          <h3>
            {pred.model_used}: {pred.prediction}
          </h3>
          <p>
            Confidence: <strong>{pred.confidence_percent}</strong>
          </p>
          <div className="prob-bar">
            <div
              className="prob-fill"
              style={{
                width: pred.confidence_percent,
                backgroundColor:
                  pred.prediction === "REAL" ? "#28a745" : "#dc3545",
              }}
            ></div>
          </div>
        </div>
      ))}
    </div>
  );
};

export default Results;
