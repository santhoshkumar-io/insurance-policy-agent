import type { Metadata } from "next";
import Link from "next/link";
import { Geist, Geist_Mono } from "next/font/google";
import "./globals.css";

const geistSans = Geist({ variable: "--font-geist-sans", subsets: ["latin"] });
const geistMono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Policy Intelligence",
  description:
    "Evidence-backed question answering over insurance policy wordings, with verified citations.",
};

const NAV = [
  { href: "/", label: "Library" },
  { href: "/upload", label: "Upload" },
  { href: "/ask", label: "Ask" },
];

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html
      lang="en"
      className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}
    >
      <body className="min-h-full flex flex-col bg-background text-foreground">
        <header className="border-b">
          <div className="mx-auto flex w-full max-w-6xl items-baseline gap-6 px-6 py-4">
            <Link href="/" className="font-heading text-base font-semibold">
              Policy Intelligence
            </Link>
            <nav className="flex gap-4 text-sm text-muted-foreground">
              {NAV.map((item) => (
                <Link key={item.href} href={item.href} className="hover:text-foreground">
                  {item.label}
                </Link>
              ))}
            </nav>
          </div>
        </header>
        <main className="mx-auto w-full max-w-6xl flex-1 px-6 py-8">{children}</main>
        <footer className="border-t">
          <div className="mx-auto w-full max-w-6xl px-6 py-4 text-xs text-muted-foreground">
            Answers are grounded in retrieved policy text with validated citations. Not a
            claims authority, and not advice — read the cited clause.
          </div>
        </footer>
      </body>
    </html>
  );
}
