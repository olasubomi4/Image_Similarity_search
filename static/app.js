require('dotenv').config();


document.getElementById('uploadButton').addEventListener('click', function() {
  document.getElementById('fileInput').click();
});

document.getElementById('findButton').addEventListener('click', async function() {
  const fileInput = document.getElementById('fileInput');
  if (!fileInput.files.length) {
    return alert("Please upload a file");
  }

  const file = fileInput.files[0];
  const formData = new FormData();
  formData.append("file", file);

  try {

    const response = await axios.post(`http://20.123.40.245:8080/similar`, formData);
    displayResults(response.data);
  } catch (error) {
    console.error("Error uploading image", error);
  }
});

function displayResults(results) {
  const resultsContainer = document.getElementById('resultsContainer');
  resultsContainer.innerHTML = ''; // Clear previous results

  if (results.length > 0) {
    results.forEach((result, idx) => {
      const card = document.createElement('div');
      card.className = 'card';

      const image = document.createElement('img');
      image.src = `http://20.123.40.245:8080${result.path}`;
      image.className = 'result-image';
      image.alt = `similar-${idx}`;

      const scoreText = document.createElement('p');
      scoreText.className = 'score-text';
      scoreText.textContent = `Score: ${result.score.toFixed(4)}`;

      card.appendChild(image);
      card.appendChild(scoreText);
      resultsContainer.appendChild(card);
    });
  }
}
