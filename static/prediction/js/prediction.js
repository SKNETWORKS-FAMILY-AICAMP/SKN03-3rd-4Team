
document.getElementById('submitBtn').addEventListener('click', function() {
    const data = {
        gender: document.getElementById('floatingSelect').value,
        seniorCitizen: document.getElementById('floatingSelect').value,
        partner: document.getElementById('floatingSelect').value,
        dependents: document.getElementById('floatingSelect').value,
        tenure: document.getElementById('exampleInputEmail1').value,
        phoneService: document.getElementById('floatingSelect').value,
        multipleLines: document.getElementById('exampleInputEmail1').value,
        internetService: document.getElementById('floatingSelect').value,
        onlineSecurity: document.getElementById('floatingSelect').value,
        onlineBackup: document.getElementById('floatingSelect').value,
        deviceProtection: document.getElementById('floatingSelect').value,
        techSupport: document.getElementById('floatingSelect').value,
        streamingTV: document.getElementById('floatingSelect').value,
        streamingMovies: document.getElementById('floatingSelect').value,
        contract: document.getElementById('floatingSelect').value,
        paperlessBilling: document.getElementById('floatingSelect').value,
        paymentMethod: document.getElementById('floatingSelect').value,
        monthlyCharges: document.getElementById('monthlyCharges').value,
        totalCharges: document.getElementById('totalCharges').value
    };

    fetch('predict_churn', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
            'X-CSRFToken': '{{ csrf_token }}' // Django CSRF token
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(result => {
        document.getElementById('result').innerText = JSON.stringify(result);
    })
    .catch(error => {
        console.error('Error:', error);
    });
});