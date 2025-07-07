// DOM element references
const uploadBox      = document.getElementById("uploadBox");
const fileInput      = document.getElementById("fileInput");
const previewImg     = document.getElementById("previewImg");
const previewVid     = document.getElementById("previewVid");
const analyzeBtn     = document.getElementById("analyzeBtn");
const loading        = document.getElementById("loading");
const resultBox      = document.getElementById("result");
const confidenceSpan = document.getElementById("confidence");
const confidenceFill = document.getElementById("confidenceFill");
const badge          = document.getElementById("badge");
const processTimeSpan= document.getElementById("processTime");
const fileSizeSpan   = document.getElementById("fileSize");
const datasetNameEl  = document.getElementById("datasetName");

let selectedFile = null;
let startTime    = null;

function formatFileSize(bytes) {
  if (bytes === 0) return '0 Bytes';
  const k = 1024;
  const sizes = ['Bytes','KB','MB','GB'];
  const i = Math.floor(Math.log(bytes)/Math.log(k));
  return `${(bytes/Math.pow(k,i)).toFixed(2)} ${sizes[i]}`;
}

// File‐picker UI
uploadBox.addEventListener("click", () => fileInput.click());
uploadBox.addEventListener("dragover", e => {
  e.preventDefault();
  uploadBox.classList.add("drag-active");
});
uploadBox.addEventListener("dragleave", e => {
  e.preventDefault();
  uploadBox.classList.remove("drag-active");
});
uploadBox.addEventListener("drop", e => {
  e.preventDefault();
  uploadBox.classList.remove("drag-active");
  if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener("change", () => {
  if (fileInput.files.length) handleFile(fileInput.files[0]);
});

function handleFile(file) {
  if (file.size > 50 * 1024 * 1024) {
    alert("File size exceeds 50 MB limit.");
    return;
  }
  selectedFile = file;
  analyzeBtn.style.display = "inline-block";
  resultBox.style.display  = "none";
  fileSizeSpan.textContent  = formatFileSize(file.size);

  const reader = new FileReader();
  reader.onload = e => {
    if (file.type.startsWith("image/")) {
      previewImg.src = e.target.result;
      previewImg.style.display = "block";
      previewVid.style.display = "none";
    } else if (file.type.startsWith("video/")) {
      previewVid.src = e.target.result;
      previewVid.style.display = "block";
      previewImg.style.display = "none";
    }
  };
  reader.readAsDataURL(file);
}

analyzeBtn.addEventListener("click", async () => {
  if (!selectedFile) return;

  startTime = performance.now();
  loading.style.display   = "flex";
  analyzeBtn.style.display= "none";
  resultBox.style.display = "none";

  const formData = new FormData();
  formData.append("file", selectedFile);

  let data;
  try {
    const res = await fetch("http://localhost:5000/analyze", {
      method: "POST",
      body: formData
    });

    // if HTTP status is not 2xx, try to extract error message
    if (!res.ok) {
      let errMsg = `HTTP ${res.status}`;
      try {
        const errJson = await res.json();
        if (errJson.error) errMsg = errJson.error;
      } catch {}
      throw new Error(errMsg);
    }

    data = await res.json();
  } catch (err) {
    loading.style.display    = "none";
    analyzeBtn.style.display = "inline-block";
    alert("Analysis failed: " + err.message);
    console.error(err);
    return;
  }

  const processTime = ((performance.now() - startTime)/1000).toFixed(2);
  loading.style.display   = "none";
  analyzeBtn.style.display= "inline-block";

  if (data.error) {
    alert("Error: " + data.error);
    return;
  }

  const conf = Number(data.confidence).toFixed(2);
  confidenceSpan.textContent = conf;
  confidenceFill.style.width = `${conf}%`;
  processTimeSpan.textContent = processTime;
  badge.textContent = data.is_real
    ? "✅ Authentic Media"
    : "⚠️ Potential Deepfake";
  badge.className = `badge ${data.is_real ? 'real' : 'fake'}`;

  resultBox.style.display = "block";
  resultBox.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
});

// Optional helper
function updateDatasetName(newName) {
  if (datasetNameEl) datasetNameEl.textContent = newName;
}
