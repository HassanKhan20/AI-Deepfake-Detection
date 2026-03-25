import React, { useState } from 'react';

interface Prediction {
  label: string;
  fake_confidence: number;
  real_confidence: number;
}

const API_BASE = 'http://localhost:8000';

const fileToBase64 = (file: File): Promise<string> =>
  new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const result = reader.result as string;
      resolve(result);
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });

const App: React.FC = () => {
  const [imageFile, setImageFile] = useState<File | null>(null);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const runPredict = async () => {
    if (!imageFile) return;
    setLoading(true);
    setError(null);
    try {
      const b64 = await fileToBase64(imageFile);
      const response = await fetch(`${API_BASE}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image_base64: b64 }),
      });
      if (!response.ok) throw new Error(`API error ${response.status}`);
      const data = await response.json();
      setPrediction(data);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className='app-container'>
      <h1>AI Deepfake Detection</h1>
      <p>Upload an image and get real/fake classification with confidence.</p>
      <input
        type='file'
        accept='image/*'
        onChange={e => {
          setPrediction(null);
          setError(null);
          setImageFile(e.target.files?.[0] ?? null);
        }}
      />
      <button onClick={runPredict} disabled={!imageFile || loading}>
        {loading ? 'Analyzing...' : 'Run Inference'}
      </button>
      {error && <div className='error'>{error}</div>}
      {prediction && (
        <div className='result-card'>
          <h2>Prediction: {prediction.label.toUpperCase()}</h2>
          <div className='bar'>
            <span>Fake</span>
            <div className='fill' style={{ width: `${(prediction.fake_confidence || 0) * 100}%` }} />
            <strong>{(prediction.fake_confidence * 100).toFixed(2)}%</strong>
          </div>
          <div className='bar'>
            <span>Real</span>
            <div className='fill real' style={{ width: `${(prediction.real_confidence || 0) * 100}%` }} />
            <strong>{(prediction.real_confidence * 100).toFixed(2)}%</strong>
          </div>
        </div>
      )}
    </div>
  );
};

export default App;
