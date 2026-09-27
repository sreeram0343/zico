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
    <html lang="en">
      <body className="bg-[#FAFAF7] text-[#101828] antialiased selection:bg-[#F7C948] selection:text-[#101828]">
        {children}
      </body>
    </html>
  );
}
