import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "DrugMicrobe AI",
  description:
    "AI-powered Drug-Microbe Interaction Prediction using HaGAT",
};

type LayoutProps = {
  children: React.ReactNode;
};

export default function RootLayout({
  children,
}: LayoutProps) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}