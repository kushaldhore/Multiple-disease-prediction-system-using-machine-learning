// script.js

// Smooth scrolling for navigation
document.querySelectorAll('a[href^="#"]').forEach(anchor => {
    anchor.addEventListener("click", function (e) {
        e.preventDefault();
        document.querySelector(this.getAttribute("href")).scrollIntoView({
            behavior: "smooth"
        });
    });
});


// script.js

document.querySelector('form').addEventListener('submit', function(event) {
    event.preventDefault();

    // Get form inputs
    const age = document.querySelector('input[name="age"]').value;
    const symptoms = document.querySelector('input[name="symptoms"]').value;

    // Validate input fields
    if (!age || !symptoms) {
        alert("Please fill out all the fields.");
        return;
    }

    // Process the form submission (e.g., show a loading indicator)
    document.querySelector(".cta-button").textContent = "Predicting...";

    // Here you can call the backend or a mock prediction function
    setTimeout(() => {
        // For now, let's just simulate a prediction result:
        let result = mockPrediction(age, symptoms);
        displayPredictionResult(result);
    }, 2000); // Simulating a delay

});

// Mock prediction function (replace with actual ML model in the future)
function mockPrediction(age, symptoms) {
    // Randomly generate a disease prediction based on age and symptoms
    let diseases = ["Heart Disease", "Diabetes", "Kidney Disease", "Cancer", "Respiratory Disorders"];
    let randomDisease = diseases[Math.floor(Math.random() * diseases.length)];

    return {
        disease: randomDisease,
        confidence: Math.random().toFixed(2) * 100
    };
}

// Function to display prediction result
function displayPredictionResult(result) {
    const predictionDiv = document.querySelector(".prediction-result");
    predictionDiv.innerHTML = `
        <h3>Prediction Result</h3>
        <p><strong>Disease:</strong> ${result.disease}</p>
        <p><strong>Confidence Level:</strong> ${result.confidence}%</p>
    `;
    predictionDiv.style.display = "block";
}

// script.js

document.querySelector('input[name="symptoms"]').addEventListener('input', function() {
    const symptoms = this.value.toLowerCase();
    const feedbackDiv = document.querySelector(".dynamic-feedback");

    // Display feedback based on symptom input
    if (symptoms.includes("fever") && symptoms.includes("cough")) {
        feedbackDiv.innerHTML = "These symptoms could indicate a respiratory condition such as Pneumonia or COVID-19. Please consult a doctor.";
    } else if (symptoms.includes("chest pain") && symptoms.includes("shortness of breath")) {
        feedbackDiv.innerHTML = "These symptoms may suggest heart disease. It is important to seek medical attention.";
    } else {
        feedbackDiv.innerHTML = "Enter symptoms to get feedback.";
    }

    feedbackDiv.style.display = symptoms ? "block" : "none";

});

// script.js

// Show loading spinner during prediction
document.querySelector('form').addEventListener('submit', function(event) {
    event.preventDefault();

    // Show loading spinner
    document.querySelector(".loading-spinner").style.display = "block";

    // Simulate prediction delay
    setTimeout(() => {
        document.querySelector(".loading-spinner").style.display = "none";  // Hide spinner
        let result = mockPrediction(age, symptoms);
        displayPredictionResult(result);
    }, 2000); // Simulated delay
});

// Function to display prediction result
function displayPredictionResult(result) {
    const predictionDiv = document.querySelector(".prediction-result");
    predictionDiv.innerHTML = `
        <h3>Prediction Result</h3>
        <p><strong>Disease:</strong> ${result.disease}</p>
        <p><strong>Confidence Level:</strong> ${result.confidence}%</p>
    `;
    predictionDiv.style.display = "block";
}
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>


// script.js

document.querySelector("#contact-form").addEventListener("submit", function(event) {
    event.preventDefault();

    // Collect form data
    const name = document.querySelector("#name").value;
    const email = document.querySelector("#email").value;
    const message = document.querySelector("#message").value;

    // Basic validation
    if (!name || !email || !message) {
        alert("Please fill in all fields.");
        return;
    }

    // Show a loading indicator or message while processing (optional)
    const button = document.querySelector("button");
    button.textContent = "Sending..."; // Change button text to "Sending"

    // Simulate form submission
    setTimeout(function() {
        // Success message
        alert("Your message has been sent successfully! We'll get back to you soon.");
        
        // Reset the form
        document.querySelector("#contact-form").reset();

        // Change button text back to default
        button.textContent = "Send Message";
    }, 2000); // Simulating network delay for demonstration
});



// script.js for prediction form

document.querySelector("#prediction-form").addEventListener("submit", function(event) {
    event.preventDefault();

    // Collect form data
    const patientName = document.querySelector("#patient-name").value;
    const age = document.querySelector("#age").value;
    const sex = document.querySelector("input[name='sex']:checked");
    const symptoms = document.querySelectorAll("input[name='symptoms']:checked");

    // Validation
    if (!patientName || !age || !sex || symptoms.length === 0) {
        alert("Please fill in all fields.");
        return;
    }

    // Simulate prediction result
    const selectedSymptoms = Array.from(symptoms).map(symptom => symptom.value).join(', ');
    const predictedDisease = simulateDiseasePrediction(selectedSymptoms);

    // Show result
    const resultText = document.querySelector("#result-text");
    resultText.textContent = `Patient: ${patientName}, Age: ${age}, Sex: ${sex.value}. Based on the symptoms (${selectedSymptoms}), we predict the following disease(s): ${predictedDisease}.`;

    // Display prediction result section
    document.querySelector("#prediction-result").style.display = "block";
});

// Simulate disease prediction based on symptoms
function simulateDiseasePrediction(symptoms) {
    if (symptoms.includes("Fever") && symptoms.includes("Cough")) {
        return "Possible Flu or COVID-19";
    } else if (symptoms.includes("Chest Pain") && symptoms.includes("Shortness of Breath")) {
        return "Possible Heart Disease";
    } else if (symptoms.includes("Fatigue") && symptoms.includes("Headache")) {
        return "Possible Migraine or Fatigue Syndrome";
    } else {
        return "Symptoms not enough for a prediction.";
    }
}

