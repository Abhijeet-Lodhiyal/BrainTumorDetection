/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      fontFamily:{
       "Bungee": ["Bungee Spice", "cursive"],
       "Nunito" : ["Nunito", "sans-serif"]
      },
    },
  },
  plugins: [],
}