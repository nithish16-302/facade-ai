import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Facade.ai | Modern Architectural Restyling",
  description: "Upload a photo of any building and reimagine its exterior with modern, state-of-the-art architectural palettes instantly using AI.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>
        <div className="bg-blob blob-1"></div>
        <div className="bg-blob blob-2"></div>
        
        <header className="nav-header">
          <div className="logo">Facade.ai</div>
          <a href="#" style={{color: 'var(--text-secondary)', textDecoration: 'none', fontSize: '0.9rem'}}>Sign in</a>
        </header>

        {children}
      </body>
    </html>
  );
}
