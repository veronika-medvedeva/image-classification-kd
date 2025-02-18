async function uploadImage() {
    let fileInput = document.getElementById("fileInput");
    if (!fileInput.files.length) {
        alert("Please select an image.");
        return;
    }

    let formData = new FormData();
    formData.append("file", fileInput.files[0]);

    let reader = new FileReader();
    reader.onload = function(e) {
        let imgElement = document.getElementById("uploadedImage");
        imgElement.src = e.target.result;
        document.getElementById("imageContainer").style.display = "block";
    };
    reader.readAsDataURL(fileInput.files[0]);

    try {
        let response = await fetch("http://127.0.0.1:5050/predict/", {
            method: "POST",
            body: formData
        });

        if (!response.ok) {
            throw new Error('Failed to get prediction from backend');
        }

        let result = await response.json();

        document.getElementById("result").innerText = `Predicted Class: ${result.predicted_class}, Confidence: ${result.confidence.toFixed(2)}`;
        document.getElementById("resultContainer").style.display = "block";
    } catch (error) {
        alert("Error: " + error.message);
    }
}
