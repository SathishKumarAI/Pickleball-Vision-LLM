import UploadDropzone from "@/components/UploadDropzone";
import { PageHeader } from "@/components/ui";

export default function UploadPage() {
  return (
    <div className="mx-auto max-w-2xl space-y-6">
      <PageHeader
        title="Analyze a match"
        sub="We detect players & the ball, track the rally, and write a coaching report."
      />
      <UploadDropzone />
      <p className="text-center text-xs text-ink/45">
        Your video is processed privately and can be deleted any time in Settings.
      </p>
    </div>
  );
}
