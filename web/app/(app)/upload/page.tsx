import UploadDropzone from "@/components/UploadDropzone";

export default function UploadPage() {
  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">Analyze a match</h1>
      <p className="text-slate-600">
        Upload a clip (up to 2 minutes, 300 MB). We detect players &amp; the ball,
        track the rally, and generate a coaching report.
      </p>
      <UploadDropzone />
    </div>
  );
}
