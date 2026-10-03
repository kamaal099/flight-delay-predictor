/* =========================================================
   FLIGHTAI FRONTEND
========================================================= */

const API_URL = "http://127.0.0.1:8000";


/* =========================================================
   DOM ELEMENTS
========================================================= */

const form =
    document.getElementById("predictionForm");

const predictButton =
    document.getElementById("predictButton");

const buttonText =
    document.getElementById("buttonText");

const resetButton =
    document.getElementById("resetButton");

const sampleButton =
    document.getElementById("sampleButton");

const emptyState =
    document.getElementById("emptyState");

const predictionContent =
    document.getElementById("predictionContent");

const resultStatus =
    document.getElementById("resultStatus");

const predictionBanner =
    document.getElementById("predictionBanner");

const predictionIcon =
    document.getElementById("predictionIcon");

const predictionTitle =
    document.getElementById("predictionTitle");

const predictionDescription =
    document.getElementById("predictionDescription");

const probabilityValue =
    document.getElementById("probabilityValue");

const progressBar =
    document.getElementById("progressBar");

const predictionMetric =
    document.getElementById("predictionMetric");

const thresholdMetric =
    document.getElementById("thresholdMetric");

const thresholdLabel =
    document.getElementById("thresholdLabel");

const routeValue =
    document.getElementById("routeValue");

const airlineValue =
    document.getElementById("airlineValue");

const apiStatus =
    document.getElementById("apiStatus");

const apiStatusText =
    document.getElementById("apiStatusText");

const toast =
    document.getElementById("toast");

const toastMessage =
    document.getElementById("toastMessage");


/* =========================================================
   API HEALTH CHECK
========================================================= */

async function checkAPI() {

    try {

        const response =
            await fetch(
                `${API_URL}/health`,
                {
                    method: "GET"
                }
            );

        if (!response.ok) {
            throw new Error("API unavailable");
        }

        apiStatus.classList.add("online");

        apiStatus.classList.remove("offline");

        apiStatusText.textContent =
            "API Online";

    }

    catch (error) {

        apiStatus.classList.add("offline");

        apiStatus.classList.remove("online");

        apiStatusText.textContent =
            "API Offline";

    }
}


/* =========================================================
   INITIAL API CHECK
========================================================= */

checkAPI();


/*
   Check API every 15 seconds.
*/

setInterval(
    checkAPI,
    15000
);


/* =========================================================
   DATE
========================================================= */

function setDefaultDate() {

    const dateInput =
        document.getElementById("flightDate");

    const today =
        new Date();

    const year =
        today.getFullYear();

    const month =
        String(
            today.getMonth() + 1
        ).padStart(2, "0");

    const day =
        String(
            today.getDate()
        ).padStart(2, "0");

    dateInput.value =
        `${year}-${month}-${day}`;
}

setDefaultDate();


/* =========================================================
   INPUT NORMALIZATION
========================================================= */

function uppercaseInput(id) {

    const input =
        document.getElementById(id);

    input.addEventListener(
        "input",
        () => {

            input.value =
                input.value
                    .toUpperCase()
                    .replace(/[^A-Z]/g, "");

        }
    );
}

uppercaseInput("airline");
uppercaseInput("origin");
uppercaseInput("destination");


/* =========================================================
   TIME CONVERSION
========================================================= */

/*
   HTML time:
   16:49

   API:
   1649
*/

function timeToHHMM(time) {

    if (!time) {
        return 0;
    }

    const parts =
        time.split(":");

    const hour =
        parseInt(parts[0], 10);

    const minute =
        parseInt(parts[1], 10);

    return (
        hour * 100 +
        minute
    );
}


/* =========================================================
   TOAST
========================================================= */

function showToast(message) {

    toastMessage.textContent =
        message;

    toast.classList.add("show");

    setTimeout(
        () => {
            toast.classList.remove("show");
        },
        3000
    );
}


/* =========================================================
   LOADING STATE
========================================================= */

function setLoading(loading) {

    if (loading) {

        predictButton.classList.add(
            "loading"
        );

        buttonText.textContent =
            "Analyzing flight...";

    }

    else {

        predictButton.classList.remove(
            "loading"
        );

        buttonText.textContent =
            "Predict Delay";

    }
}


/* =========================================================
   BUILD API PAYLOAD
========================================================= */

function buildPayload() {

    const flightDate =
        document.getElementById(
            "flightDate"
        ).value;

    const airline =
        document.getElementById(
            "airline"
        ).value.trim();

    const origin =
        document.getElementById(
            "origin"
        ).value.trim();

    const destination =
        document.getElementById(
            "destination"
        ).value.trim();

    const departure =
        document.getElementById(
            "departure"
        ).value;

    const arrival =
        document.getElementById(
            "arrival"
        ).value;

    const duration =
        document.getElementById(
            "duration"
        ).value;

    const distance =
        document.getElementById(
            "distance"
        ).value;


    return {

        FlightDate:
            flightDate,

        Reporting_Airline:
            airline,

        Origin:
            origin,

        Dest:
            destination,

        CRSDepTime:
            timeToHHMM(departure),

        CRSArrTime:
            timeToHHMM(arrival),

        CRSElapsedTime:
            Number(duration),

        Distance:
            Number(distance)

    };
}


/* =========================================================
   DISPLAY RESULT
========================================================= */

function displayResult(result, payload) {

    const probability =
        Number(
            result.delay_probability
        );

    const threshold =
        Number(
            result.threshold ?? 0.38
        );

    const prediction =
        Number(
            result.prediction
        );

    const percentage =
        probability * 100;


    /* Hide empty state */

    emptyState.style.display =
        "none";

    predictionContent.classList.add(
        "visible"
    );


    /* Status */

    resultStatus.textContent =
        "COMPLETE";

    resultStatus.classList.add(
        "complete"
    );


    /* Probability */

    animateProbability(
        percentage
    );


    /* Threshold */

    thresholdMetric.textContent =
        threshold.toFixed(2);

    thresholdLabel.textContent =
        `Threshold ${Math.round(
            threshold * 100
        )}%`;


    /* Prediction */

    if (prediction === 1) {

        predictionTitle.textContent =
            "Flight Delayed";

        predictionDescription.textContent =
            "The model predicts a higher probability of delay.";

        predictionIcon.textContent =
            "!";

        predictionBanner.classList.remove(
            "on-time"
        );

        predictionMetric.textContent =
            "Delayed";

    }

    else {

        predictionTitle.textContent =
            "Flight On Time";

        predictionDescription.textContent =
            "The model predicts a lower probability of delay.";

        predictionIcon.textContent =
            "✓";

        predictionBanner.classList.add(
            "on-time"
        );

        predictionMetric.textContent =
            "On Time";

    }


    /* Route */

    routeValue.textContent =
        `${payload.Origin} → ${payload.Dest}`;

    airlineValue.textContent =
        payload.Reporting_Airline;


    /* Scroll result into view */

    if (window.innerWidth < 900) {

        document
            .querySelector(".result-card")
            .scrollIntoView({
                behavior: "smooth",
                block: "center"
            });

    }
}


/* =========================================================
   PROBABILITY ANIMATION
========================================================= */

function animateProbability(
    target
) {

    let current = 0;

    const duration = 900;

    const start =
        performance.now();

    function update(now) {

        const elapsed =
            now - start;

        const progress =
            Math.min(
                elapsed / duration,
                1
            );

        /*
           Smooth ease-out
        */

        const eased =
            1 -
            Math.pow(
                1 - progress,
                3
            );

        current =
            target * eased;

        probabilityValue.textContent =
            `${current.toFixed(1)}%`;

        progressBar.style.width =
            `${current}%`;


        if (progress < 1) {

            requestAnimationFrame(
                update
            );

        }

    }

    requestAnimationFrame(
        update
    );
}


/* =========================================================
   PREDICT
========================================================= */

form.addEventListener(
    "submit",
    async function(event) {

        event.preventDefault();


        const payload =
            buildPayload();


        /*
           Basic validation
        */

        if (
            !payload.FlightDate ||
            !payload.Reporting_Airline ||
            !payload.Origin ||
            !payload.Dest ||
            !payload.CRSDepTime ||
            !payload.CRSArrTime ||
            !payload.CRSElapsedTime ||
            !payload.Distance
        ) {

            showToast(
                "Please complete all flight details."
            );

            return;
        }


        setLoading(true);


        try {

            const response =
                await fetch(
                    `${API_URL}/predict`,
                    {
                        method: "POST",

                        headers: {
                            "Content-Type":
                                "application/json"
                        },

                        body:
                            JSON.stringify(
                                payload
                            )
                    }
                );


            /*
               API error
            */

            if (!response.ok) {

                let errorMessage =
                    "Prediction failed.";

                try {

                    const errorData =
                        await response.json();

                    if (
                        errorData.detail
                    ) {

                        errorMessage =
                            errorData.detail;

                    }

                }

                catch (_) {}

                throw new Error(
                    errorMessage
                );
            }


            const result =
                await response.json();


            console.log(
                "Prediction:",
                result
            );


            displayResult(
                result,
                payload
            );


            showToast(
                "Prediction completed successfully."
            );

        }

        catch (error) {

            console.error(
                error
            );

            showToast(
                error.message ||
                "Unable to connect to the prediction API."
            );

        }

        finally {

            setLoading(false);

        }

    }
);


/* =========================================================
   SAMPLE FLIGHT
========================================================= */

sampleButton.addEventListener(
    "click",
    () => {

        document.getElementById(
            "flightDate"
        ).value =
            "2026-07-15";

        document.getElementById(
            "airline"
        ).value =
            "DL";

        document.getElementById(
            "origin"
        ).value =
            "DTW";

        document.getElementById(
            "destination"
        ).value =
            "MKE";

        document.getElementById(
            "departure"
        ).value =
            "16:49";

        document.getElementById(
            "arrival"
        ).value =
            "17:50";

        document.getElementById(
            "duration"
        ).value =
            "72";

        document.getElementById(
            "distance"
        ).value =
            "237";


        showToast(
            "Sample flight loaded."
        );

    }
);


/* =========================================================
   RESET
========================================================= */

resetButton.addEventListener(
    "click",
    () => {

        form.reset();

        setDefaultDate();


        emptyState.style.display =
            "flex";

        predictionContent.classList.remove(
            "visible"
        );


        resultStatus.textContent =
            "READY";

        resultStatus.classList.remove(
            "complete"
        );


        progressBar.style.width =
            "0%";

        probabilityValue.textContent =
            "0.0%";


        predictionBanner.classList.remove(
            "on-time"
        );


        predictionMetric.textContent =
            "—";

        thresholdMetric.textContent =
            "0.38";

        routeValue.textContent =
            "DTW → MKE";

        airlineValue.textContent =
            "DL";


        showToast(
            "Prediction reset."
        );

    }
);