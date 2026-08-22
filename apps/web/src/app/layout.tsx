import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: 'ZICO | Intelligent Travel Operations & Orchestration',
  description: 'Real-time multi-agent autonomous travel orchestration with human-in-the-loop verification and live streaming.',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className="bg-slate-950 text-slate-50 antialiased selection:bg-blue-600 selection:text-white">
        {children}
      </body>
    </html>
  );
}
