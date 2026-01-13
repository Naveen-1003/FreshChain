/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        freshchain: {
          primary: '#0D9488',      // Teal (Primary Brand)
          secondary: '#F0FDFA',    // Very Light Teal (Cards Background)
          accent: '#14B8A6',       // Bright Teal (Interactive)
          light: '#FFFFFF',        // White (Main Background)
          dark: '#134E4A',         // Dark Teal (Text)
          warning: '#F59E0B',      // Amber (Warning)
          danger: '#EF4444',       // Red (Error)
          success: '#10B981',      // Emerald (Success)
        },
        slate: {
          50: '#f8fafc',
          100: '#f1f5f9',
          200: '#e2e8f0',
          300: '#cbd5e1',
          400: '#94a3b8',
          500: '#64748b',
          600: '#475569',
          700: '#334155',
          800: '#1e293b',
          900: '#0f172a',
        },
        gray: {
          50: '#F3F5F4',           // Light Grey
          100: '#F3F5F4',
          200: '#e5e7eb',
          300: '#d1d5db',
          400: '#9ca3af',
          500: '#6B7280',          // Medium Grey (Text / Labels)
          600: '#4b5563',
          700: '#374151',
          800: '#1f2937',
          900: '#111827',
        },
        emerald: {
          25: '#f0fdf4',
          50: '#ecfdf5',
          100: '#d1fae5',
          200: '#a7f3d0',
          300: '#6ee7b7',
          400: '#34d399',
          500: '#10b981',
          600: '#059669',
          700: '#047857',
          800: '#065f46',
          900: '#064e3b',
        },
        teal: {
          25: '#f0fdfa',
          50: '#f0fdfa',
          100: '#ccfbf1',
          200: '#99f6e4',
          300: '#5eead4',
          400: '#2dd4bf',
          500: '#14b8a6',
          600: '#0d9488',
          700: '#0f766e',
          800: '#115e59',
          900: '#134e4a',
        }
      },
      backgroundImage: {
        'freshchain-gradient': 'linear-gradient(135deg, #1FA97A 0%, #0F6E55 100%)', // 🌊 Gradient Background
        'hero-gradient': 'linear-gradient(135deg, #1FA97A 0%, #0F6E55 100%)',
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
      },
      boxShadow: {
        'soft': '0 2px 15px -3px rgba(31, 169, 122, 0.1), 0 10px 20px -2px rgba(31, 169, 122, 0.05)',
        'green': '0 4px 14px 0 rgba(31, 169, 122, 0.15)',
      }
    },
  },
  plugins: [],
}