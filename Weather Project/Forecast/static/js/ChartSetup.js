document.addEventListener('DOMContentLoaded', () => {
    const chartElement = document.getElementById('chart');
    if (!chartElement) {
        console.error('Chart element not found');
        return;
    }

 
    const ctx = chartElement.getContext('2d');
    const gradient = ctx.createLinearGradient(0, 0, 0, 300);
    gradient.addColorStop(0, 'rgba(250, 0, 0, 1)');
    gradient.addColorStop(1, 'rgba(136, 255, 0, 1)');

    const forecastItems = document.querySelectorAll('.forecast__item');

    const temps = [];
    const times = [];

    forecastItems.forEach(item => {
        const time = item.querySelector('.forecast-time')?.textContent;
        const temp = item.querySelector('.forecast-tempertureValue')?.textContent;

        if (time && temp) {
            times.push(time);
            temps.push(parseFloat(temp));
        }
    });

    if (temps.length === 0 || times.length === 0) {
        console.error('No valid temperature or time data found');
        return;
    }

    new Chart(ctx, {
        type: 'line',
        data: {
            labels: times,
            datasets: [{
                label: 'Celsius Degrees',
                data: temps,
                borderColor: gradient,
                borderWidth: 2,
                tension: 0.4,
                pointRadius: 4,
                fill: false
            }],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                x: { display: false },
                y: { display: false }
            },
            animation: { duration: 750 }
        },
    });
});
