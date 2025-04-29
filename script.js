document.addEventListener("DOMContentLoaded", () => {
    // Hamburger Menu Toggle
    const hamburger = document.querySelector('.hamburger');
    const navLinks = document.querySelector('.nav-links');

    hamburger.addEventListener('click', () => {
        navLinks.classList.toggle('show');
    });

    // Get DOM elements
    const fraudForm = document.getElementById("fraudForm");
    const loadingIndicator = document.getElementById("loadingIndicator");
    const resultContainer = document.getElementById("resultContainer");
    
    // Set current time in form
    const now = new Date();
    document.getElementById("hour").value = now.getHours();
    document.getElementById("day").value = now.getDay() === 0 ? 6 : now.getDay() - 1; // Adjust Sunday

    // Fraud Detection Form Handler
    fraudForm.addEventListener("submit", async function(event) {
        event.preventDefault();
        
        // Show loading indicator
        loadingIndicator.style.display = "flex";
        resultContainer.style.display = "none";
        
        // Get form values
        const amount = document.getElementById("amount").value;
        const location = document.getElementById("location").value;
        const transactionType = document.getElementById("transactionType").value;
        const hour = document.getElementById("hour").value || now.getHours();
        const day = document.getElementById("day").value || (now.getDay() === 0 ? 6 : now.getDay() - 1);

        // Prepare request data
        const requestData = {
            amount: amount,
            location: location,
            transaction_type: transactionType,
            hour: parseInt(hour),
            day: parseInt(day)
        };

        try {
            // Send request to API
            const response = await fetch("http://127.0.0.1:5050/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(requestData)
            });

            // Hide loading indicator
            loadingIndicator.style.display = "none";
            
            // Process response
            const data = await response.json();
            
            if (data.error) {
                showError(data.error);
                return;
            }
            
            // Display results
            displayResults(data, requestData);
            
        } catch (error) {
            // Hide loading indicator
            loadingIndicator.style.display = "none";
            
            console.error("Error:", error);
            showError("Server connection failed!");
        }
    });
    
    // Function to display error
    function showError(message) {
        resultContainer.style.display = "block";
        document.getElementById("resultTitle").innerText = "Error";
        document.getElementById("resultBadge").innerText = "Error";
        document.getElementById("resultBadge").style.backgroundColor = "#ff6464";
        document.getElementById("riskLevel").innerText = "N/A";
        document.getElementById("confidence").innerText = "N/A";
        document.getElementById("riskGaugeFill").style.width = "0%";
        document.getElementById("riskFactorsContainer").style.display = "none";
        document.getElementById("recommendationsContainer").style.display = "none";
    }
    
    // Function to display results
    function displayResults(data, requestData) {
        resultContainer.style.display = "block";
        
        // Set result title and badge
        document.getElementById("resultTitle").innerText = data.prediction;
        document.getElementById("resultBadge").innerText = data.risk_level;
        document.getElementById("resultBadge").style.backgroundColor = getColorForRiskLevel(data.color);
        
        // Set risk level and confidence
        document.getElementById("riskLevel").innerText = data.risk_level;
        document.getElementById("confidence").innerText = `${data.confidence}%`;
        
        // Update risk gauge
        document.getElementById("riskGaugeFill").style.width = `${data.confidence}%`;
        document.getElementById("riskGaugeFill").style.backgroundColor = getColorForRiskLevel(data.color);
        
        // Display risk factors
        const riskFactorsList = document.getElementById("riskFactors");
        riskFactorsList.innerHTML = "";
        
        if (data.risk_factors && data.risk_factors.length > 0) {
            document.getElementById("riskFactorsContainer").style.display = "block";
            data.risk_factors.forEach(factor => {
                const li = document.createElement("li");
                li.textContent = factor;
                riskFactorsList.appendChild(li);
            });
        } else {
            document.getElementById("riskFactorsContainer").style.display = "none";
        }
        
        // Display recommendations
        const recommendationsList = document.getElementById("recommendations");
        recommendationsList.innerHTML = "";
        
        if (data.recommendations && data.recommendations.length > 0) {
            document.getElementById("recommendationsContainer").style.display = "block";
            data.recommendations.forEach(recommendation => {
                const li = document.createElement("li");
                li.textContent = recommendation;
                recommendationsList.appendChild(li);
            });
        } else {
            document.getElementById("recommendationsContainer").style.display = "none";
        }
    }
    
    // Helper function to get color for risk level
    function getColorForRiskLevel(color) {
        switch(color) {
            case "green": return "#64ffda";
            case "orange": return "#ffb347";
            case "red": return "#ff6464";
            default: return "#64ffda";
        }
    }
});