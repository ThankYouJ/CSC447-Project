function chooseFile() {
    document.getElementById("fileInput").click();
}

let selectedFile = null;

function previewImage(event) {
    selectedFile = event.target.files[0];
    let preview = document.getElementById("preview");
    let uploadText = document.getElementById("uploadText");

    preview.src = URL.createObjectURL(selectedFile);
    preview.style.display = "block";
    uploadText.style.display = "none";
}

async function analyze() {
    if (!selectedFile) {
        alert("Please upload a leaf image first!");
        return;
    }

    let formData = new FormData();
    formData.append("file", selectedFile);

    let res = await fetch("/predict", {
        method: "POST",
        body: formData
    });

    let data = await res.json();

    let txt = `Prediction: ${data.predicted}\n\nProbabilities:\n`;
    for (let key in data.probabilities) {
        txt += `${key}: ${(data.probabilities[key] * 100).toFixed(1)}%\n`;
    }

    document.getElementById("result").innerText = txt;
}
