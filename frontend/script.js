document.getElementById('prediction-form').addEventListener('submit', async (e) => {
    e.preventDefault();
    
    const btnText = document.getElementById('btn-text');
    const btnSpinner = document.getElementById('btn-spinner');
    const submitBtn = document.getElementById('submit-btn');
    const resultContainer = document.getElementById('result-container');
    const errorContainer = document.getElementById('error-container');
    
    // UI Loading state
    btnText.textContent = "Generating prediction...";
    btnSpinner.classList.remove('hidden');
    submitBtn.disabled = true;
    submitBtn.classList.add('opacity-90', 'cursor-not-allowed');
    resultContainer.classList.add('hidden');
    errorContainer.classList.add('hidden');

    const getBool = (id) => document.getElementById(id).checked ? 1.0 : 0.0;
    
    const payload = {
        age: parseInt(document.getElementById('age').value),
        sex: document.getElementById('sex').value,
        state: document.getElementById('state').value,
        dependents: parseInt(document.getElementById('dependents').value),
        income: parseFloat(document.getElementById('income').value),
        bmi: parseFloat(document.getElementById('bmi').value),
        smoker: document.getElementById('smoker').value,
        systolic_bp: parseFloat(document.getElementById('systolic_bp').value),
        diastolic_bp: parseFloat(document.getElementById('diastolic_bp').value),
        
        urban_rural: document.getElementById('urban_rural').value,
        education: document.getElementById('education').value,
        marital_status: document.getElementById('marital_status').value,
        employment_status: document.getElementById('employment_status').value,
        alcohol_freq: document.getElementById('alcohol_freq').value,
        
        visits_last_year: parseFloat(document.getElementById('visits_last_year').value),
        hospitalizations_last_3yrs: parseFloat(document.getElementById('hospitalizations_last_3yrs').value),
        days_hospitalized_last_3yrs: parseFloat(document.getElementById('days_hospitalized_last_3yrs').value),
        medication_count: parseFloat(document.getElementById('medication_count').value),
        
        ldl: parseFloat(document.getElementById('ldl').value),
        hba1c: parseFloat(document.getElementById('hba1c').value),
        
        hypertension: getBool('hypertension'),
        diabetes: getBool('diabetes'),
        is_high_risk: getBool('is_high_risk'),
        asthma: getBool('asthma'),
        copd: getBool('copd'),
        cardiovascular_disease: getBool('cardiovascular_disease'),
        cancer_history: getBool('cancer_history'),
        kidney_disease: getBool('kidney_disease'),
        liver_disease: getBool('liver_disease'),
        arthritis: getBool('arthritis'),
        mental_health: getBool('mental_health')
    };

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });

        const data = await response.json();

        if (response.ok) {
            // Use en-IN locale for Indian comma formatting (e.g. ₹2,11,072)
            const formatter = new Intl.NumberFormat('en-IN', { 
                style: 'currency', 
                currency: 'INR',
                maximumFractionDigits: 0
            });
            
            document.getElementById('prediction-result').textContent = formatter.format(data.predicted_cost);
            resultContainer.classList.remove('hidden');
            
            // Scroll to result smoothly
            resultContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        } else {
            throw new Error(data.detail || "Failed to generate prediction. Check your inputs.");
        }
    } catch (error) {
        document.getElementById('error-message').textContent = "Error: " + error.message;
        errorContainer.classList.remove('hidden');
    } finally {
        btnText.textContent = "GENERATE PREDICTION";
        btnSpinner.classList.add('hidden');
        submitBtn.disabled = false;
        submitBtn.classList.remove('opacity-90', 'cursor-not-allowed');
    }
});

document.getElementById('prediction-form').addEventListener('reset', () => {
    document.getElementById('result-container').classList.add('hidden');
    document.getElementById('error-container').classList.add('hidden');
});

// Theme Toggle Logic
document.addEventListener('DOMContentLoaded', () => {
    const themeToggle = document.getElementById('theme-toggle');
    const currentTheme = document.documentElement.getAttribute('data-theme');
    
    // Set initial toggle state based on html attribute (which is set by inline script in head)
    if (currentTheme === 'dark') {
        themeToggle.checked = true;
    }

    themeToggle.addEventListener('change', (e) => {
        if (e.target.checked) {
            document.documentElement.setAttribute('data-theme', 'dark');
            localStorage.setItem('mica-theme', 'dark');
        } else {
            document.documentElement.removeAttribute('data-theme');
            localStorage.setItem('mica-theme', 'light');
        }
    });
});
