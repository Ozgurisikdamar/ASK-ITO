import { BrowserRouter, Routes, Route, Link } from "react-router-dom";
import { Home } from "./pages/Home";
import { Results } from "./pages/Results";
import { EvalReview } from "./pages/EvalReview";

function EvalPlaceholder() {
  return (
    <div className="min-h-screen bg-surface p-8 text-content">
      <Link to="/" className="text-link hover:underline">← Ask ITO</Link>
      <h1 className="mt-4 text-xl font-bold">Evaluation panel</h1>
      <p className="mt-2 text-content-muted">
        The evaluation panel (ablation table, charts, run comparison) is built in
        milestone M8. The{" "}
        <Link to="/eval/review" className="text-link hover:underline">test set review</Link>{" "}
        page is available now.
      </p>
    </div>
  );
}

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/search" element={<Results />} />
        <Route path="/eval" element={<EvalPlaceholder />} />
        <Route path="/eval/review" element={<EvalReview />} />
      </Routes>
    </BrowserRouter>
  );
}
