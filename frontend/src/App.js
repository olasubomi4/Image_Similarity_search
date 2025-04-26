import React, { useState } from 'react';
import axios from 'axios';

function App() {
  const [file, setFile] = useState(null);
  const [results, setResults] = useState([]);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setResults([]);
  };

  const handleUpload = async () => {
    if (!file) return alert("Please upload a file");

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await axios.post("http://127.0.0.1:5000/similar", formData);
      setResults(res.data);
    } catch (error) {
      console.error("Error uploading image", error);
    }
  };

  const handleFileButtonClick = () => {
    document.getElementById('file-input').click(); // Triggers the hidden file input when the button is clicked
  };

  return (
    <div style={styles.container}>
      <h2 style={styles.heading}>Image Similarity Search</h2>
      <div style={styles.uploadContainer}>
        <button
          onClick={handleFileButtonClick}
          style={styles.uploadButton}
        >
          Upload Fruit Image
        </button>
        <input
          id="file-input"
          type="file"
          onChange={handleFileChange}
          style={styles.fileInput}
        />
        <br /><br />
        <button
          onClick={handleUpload}
          style={styles.uploadButton}
        >
          Find Similar fruits.
        </button>
      </div>

      <div style={styles.resultsContainer}>
        {results.length > 0 && results.map((result, idx) => (
          <div key={idx} style={styles.card}>
            <img
              src={`http://127.0.0.1:5000${result.path}`}
              alt={`similar-${idx}`}
              style={styles.resultImage}
            />
            <p style={styles.scoreText}>Score: {result.score.toFixed(4)}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

const styles = {
  container: {
    textAlign: 'center',
    padding: '40px',
    fontFamily: 'Arial, sans-serif',
    backgroundColor: '#f7f7f7',
    borderRadius: '10px',
    maxWidth: '800px',
    margin: 'auto',
    boxShadow: '0 4px 8px rgba(0, 0, 0, 0.1)',
  },
  heading: {
    fontSize: '2rem',
    marginBottom: '20px',
    color: '#333',
  },
  uploadContainer: {
    marginBottom: '20px',
  },
  fileInput: {
    display: 'none', // Hides the default file input
  },
  uploadButton: {
    padding: '10px 20px',
    fontSize: '16px',
    backgroundColor: '#4CAF50',
    color: 'white',
    border: 'none',
    borderRadius: '5px',
    cursor: 'pointer',
    transition: 'background-color 0.3s',
  },
  uploadButtonHover: {
    backgroundColor: '#45a049',
  },
  resultsContainer: {
    marginTop: '40px',
    display: 'flex',
    flexWrap: 'wrap',
    justifyContent: 'center',
    gap: '20px',
  },
  card: {
    textAlign: 'center',
    backgroundColor: '#fff',
    padding: '10px',
    borderRadius: '8px',
    boxShadow: '0 4px 8px rgba(0, 0, 0, 0.1)',
    width: '150px',
  },
  resultImage: {
    width: '100%',
    height: '150px',
    objectFit: 'cover',
    borderRadius: '8px',
  },
  scoreText: {
    marginTop: '10px',
    color: '#555',
  },
};

export default App;
