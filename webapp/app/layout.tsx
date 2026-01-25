import type { Metadata } from "next";
import { Inter } from "next/font/google";
import "./globals.css";
import { Sidebar } from "@/components/layout/sidebar";

const inter = Inter({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Stockpile Web - AI Video Creation",
  description: "Create stunning videos with AI-powered tools",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className={inter.className}>
        <div className="flex min-h-screen">
          <Sidebar />
          <main className="ml-64 flex-1">{children}</main>
        </div>
      </body>
    </html>
  );
}
