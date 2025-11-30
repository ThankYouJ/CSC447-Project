function chooseFile() {
    document.getElementById("fileInput").click();
}

let selectedFile = null;

function previewImage(event) {
    selectedFile = event.target.files[0];

    let preview = document.getElementById("preview");
    let uploadText = document.getElementById("uploadText");

    preview.src = URL.createObjectURL(selectedFile);

    preview.classList.remove("hidden");
    uploadText.classList.add("hidden");
}

async function analyze() {
    if (!selectedFile) {
        alert("Please upload an image first!");
        return;
    }

    const formData = new FormData();
    formData.append("file", selectedFile);

    const res = await fetch("/predict", {
        method: "POST",
        body: formData
    });

    const data = await res.json();

    document.getElementById("resultCard").classList.remove("hidden");
    document.getElementById("resultTitle").innerText =
        "Prediction Result : " + data.predicted;


    const probBlock = document.getElementById("probBlock");
    probBlock.innerHTML = "";

    for (const [cls, prob] of Object.entries(data.probabilities)) {
        const p = document.createElement("p");
        p.innerText = `${cls}: ${(prob * 100).toFixed(1)}%`;
        probBlock.appendChild(p);
    }
}

function toggleProb() {
    const wrapper = document.getElementById("probWrapper");
    const btn = document.querySelector(".toggle-btn");

    if (wrapper.classList.contains("hidden")) {
        wrapper.classList.remove("hidden");
        btn.innerText = "Hide Possibilities ▲";
    } else {
        wrapper.classList.add("hidden");
        btn.innerText = "Show Possibilities ▼";
    }
}
