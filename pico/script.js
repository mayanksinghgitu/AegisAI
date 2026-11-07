document.getElementById('loginForm').addEventListener('submit', function (e) {
    e.preventDefault();


    const username = document.getElementById('username').value.trim();
    const password = document.getElementById('password').value.trim();
    const errorMsg = document.getElementById('errorMsg');

    // Clear previous error message
    errorMsg.textContent = '';

    // Simple validation
    if (!username || !password) {
        errorMsg.textContent = 'Both fields are required!';
        return;
    }

    // Example authentication
    if (username === 'mayank' && password === '2025') {
        alert('Login successful!');
        alert('Welcome MAYANK SINGH');
        // Redirect or further processing here
        window.location.href = "index.html"; // Replace with your actual redirect page
    } else {
        errorMsg.textContent = 'Invalid username or password!';
    }
});
