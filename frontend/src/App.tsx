import { BrowserRouter, Routes, Route } from "react-router-dom";
import { Home } from "./pages/Home";
import { Results } from "./pages/Results";
import { EvalReview } from "./pages/EvalReview";
import { EvalPanel } from "./pages/EvalPanel";
import { RunDetail } from "./pages/RunDetail";
import { Compare } from "./pages/Compare";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/search" element={<Results />} />
        <Route path="/eval" element={<EvalPanel />} />
        <Route path="/eval/review" element={<EvalReview />} />
        <Route path="/eval/runs/:id" element={<RunDetail />} />
        <Route path="/eval/compare/:a/:b" element={<Compare />} />
      </Routes>
    </BrowserRouter>
  );
}
