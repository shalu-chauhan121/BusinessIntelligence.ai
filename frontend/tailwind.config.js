/** @type {import('tailwindcss').Config} */
export default {
  content: ['./index.html', './src/**/*.{js,jsx}'],
  darkMode: ['class', '[data-theme="dark"]'],
  theme: {
    extend: {
      colors: {
        surface: 'var(--surface-1)',
        plane: 'var(--plane)',
        ink: {
          DEFAULT: 'var(--text-primary)',
          secondary: 'var(--text-secondary)',
          muted: 'var(--text-muted)',
        },
        hairline: 'var(--gridline)',
        edge: 'var(--baseline)',
        series: {
          1: 'var(--series-1)', 2: 'var(--series-2)', 3: 'var(--series-3)',
          4: 'var(--series-4)', 5: 'var(--series-5)',
        },
        status: {
          good: 'var(--status-good)',
          warning: 'var(--status-warning)',
          serious: 'var(--status-serious)',
          critical: 'var(--status-critical)',
        },
      },
      fontFamily: {
        sans: ['system-ui', '-apple-system', 'Segoe UI', 'sans-serif'],
      },
      borderRadius: { card: '10px' },
    },
  },
  plugins: [],
}
