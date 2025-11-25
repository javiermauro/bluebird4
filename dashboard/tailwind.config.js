/** @type {import('tailwindcss').Config} */
export default {
    content: [
        "./index.html",
        "./src/**/*.{js,ts,jsx,tsx}",
    ],
    theme: {
        extend: {
            colors: {
                background: "#050505",
                surface: "#121212",
                primary: "#6366f1", // Indigo 500
                secondary: "#a855f7", // Purple 500
                success: "#22c55e",
                danger: "#ef4444",
                warning: "#eab308",
            },
            fontFamily: {
                mono: ['"JetBrains Mono"', 'monospace'],
                sans: ['Inter', 'sans-serif'],
            }
        },
    },
    plugins: [],
}
