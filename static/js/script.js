/**
 * CardioPredict: Heart Disease Prediction and Analytics Web Platform
 * Client-Side JavaScript Logic & Preset Demo Case Loader
 * Author: Senior Data Scientist & Python Mentor
 */

document.addEventListener("DOMContentLoaded", function () {
    // Initialize all Bootstrap tooltips if present
    var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tooltipTriggerList.map(function (tooltipTriggerEl) {
        return new bootstrap.Tooltip(tooltipTriggerEl);
    });

    // Animate confidence meter fill smoothly
    const confidenceBar = document.getElementById("confidence-bar");
    if (confidenceBar) {
        const targetWidth = confidenceBar.getAttribute("data-width");
        setTimeout(() => {
            confidenceBar.style.width = targetWidth + "%";
        }, 200);
    }
});

/**
 * Pre-populate the Clinical Prediction form with verified clinical profiles.
 * Used for instant demonstrations during college project reviews.
 * 
 * @param {string} profileType - 'healthy' or 'high_risk'
 */
function loadDemoCase(profileType) {
    if (profileType === "healthy") {
        document.getElementById("Age").value = "45";
        document.getElementById("Sex").value = "0"; // Female
        document.getElementById("ChestPainType").value = "2"; // Atypical Angina
        document.getElementById("RestingBP").value = "115";
        document.getElementById("Cholesterol").value = "192";
        document.getElementById("FastingBS").value = "0"; // Normal
        document.getElementById("RestingECG").value = "0"; // Normal
        document.getElementById("MaxHR").value = "175";
        document.getElementById("ExerciseAngina").value = "0"; // No
        document.getElementById("Oldpeak").value = "0.0";
        document.getElementById("ST_Slope").value = "1"; // Upsloping
        document.getElementById("CA").value = "0";
        document.getElementById("Thal").value = "3"; // Normal
        showToast("Loaded Sample Healthy Patient profile (45 y/o Female, normal vitals).");
    } else if (profileType === "high_risk") {
        document.getElementById("Age").value = "67";
        document.getElementById("Sex").value = "1"; // Male
        document.getElementById("ChestPainType").value = "4"; // Asymptomatic
        document.getElementById("RestingBP").value = "160";
        document.getElementById("Cholesterol").value = "286";
        document.getElementById("FastingBS").value = "1"; // Sugar > 120
        document.getElementById("RestingECG").value = "2"; // LV Hypertrophy
        document.getElementById("MaxHR").value = "108";
        document.getElementById("ExerciseAngina").value = "1"; // Yes
        document.getElementById("Oldpeak").value = "3.2";
        document.getElementById("ST_Slope").value = "2"; // Flat
        document.getElementById("CA").value = "2"; // 2 vessels calcified
        document.getElementById("Thal").value = "7"; // Reversible defect
        showToast("Loaded Sample High-Risk Cardiac Patient profile (67 y/o Male, multi-vessel).");
    }
}

/**
 * Display a non-intrusive notification toast or alert banner
 */
function showToast(message) {
    const alertBox = document.getElementById("demo-alert");
    if (alertBox) {
        alertBox.innerHTML = `<i class="bi bi-info-circle-fill me-2"></i> ${message}`;
        alertBox.classList.remove("d-none");
        setTimeout(() => {
            alertBox.classList.add("d-none");
        }, 4000);
    }
}
