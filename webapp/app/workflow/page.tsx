"use client";

import { useEffect } from "react";
import { useRouter } from "next/navigation";

export default function WorkflowPage() {
  const router = useRouter();

  useEffect(() => {
    router.replace("/workflow/script");
  }, [router]);

  return (
    <div className="flex h-full items-center justify-center">
      <p className="text-muted-foreground">Redirecting to Script Writer...</p>
    </div>
  );
}
