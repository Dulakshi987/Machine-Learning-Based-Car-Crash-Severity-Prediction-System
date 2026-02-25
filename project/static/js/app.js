document.getElementById("predictForm").addEventListener("submit", async e => {
    e.preventDefault();

    const data = {
        "Crash Speed (km/h)": Number(document.getElementById("speed").value),
        "Impact Angle (degrees)": Number(document.getElementById("angle").value),
        "Airbag Deployed": document.getElementById("airbag").value,
        "Seatbelt Used": document.getElementById("seatbelt").value,
        "Weather Conditions": document.getElementById("weather").value,
        "Road Conditions": document.getElementById("road").value,
        "Crash Type": document.getElementById("crash_type").value,

        "Vehicle Type": document.getElementById("vehicle_type").value,
        "Vehicle Age (years)": Number(document.getElementById("vehicle_age").value),
        "Brake Condition": document.getElementById("brake").value,
        "Tire Condition": document.getElementById("tire").value,

        "Driver Age": Number(document.getElementById("driver_age").value),
        "Driver Experience (years)": Number(document.getElementById("experience").value),
        "Alcohol Level (BAC%)": Number(document.getElementById("alcohol").value),

        "Time of Day": document.getElementById("time").value,
        "Traffic Density": document.getElementById("traffic").value,
        "Visibility Distance (m)": Number(document.getElementById("visibility").value)
    };

    const res = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(data)
    });

    const result = await res.json();

    document.getElementById("result").innerText =
        "Severity: " + result.prediction;

    document.getElementById("confidence").innerText =
        "Confidence: " + result.confidence;

    document.getElementById("popup").style.display = "block";
});

document.getElementById("close").onclick = () => {
    document.getElementById("popup").style.display = "none";
};
