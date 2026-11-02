import { strings } from "../strings";

/** Shown with Search results: answer generation is an explicit, long-running choice. */
export function AnswerPrompt({
  onGenerate,
  autoAnswer,
  onAutoAnswerChange,
  disabled,
}: {
  onGenerate: () => void;
  autoAnswer: boolean;
  onAutoAnswerChange: (on: boolean) => void;
  disabled?: boolean;
}) {
  return (
    <section
      aria-label={strings.results.answerPromptTitle}
      className="rounded-2xl border border-mauve-200 bg-surface-raised p-4"
    >
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="min-w-0">
          <p className="text-sm font-medium text-content">{strings.results.answerPromptTitle}</p>
          <p className="text-xs text-content-muted">{strings.results.answerPromptNote}</p>
        </div>
        <button
          onClick={onGenerate}
          disabled={disabled}
          className="rounded-full bg-brand-700 px-4 py-2 text-sm font-medium text-white hover:bg-brand-500 disabled:opacity-50"
        >
          {strings.results.generateAnswer}
        </button>
      </div>
      <label className="mt-3 flex items-center gap-2 text-xs text-content-muted">
        <input
          type="checkbox"
          checked={autoAnswer}
          onChange={(e) => onAutoAnswerChange(e.target.checked)}
        />
        {strings.results.alwaysGenerate}
      </label>
    </section>
  );
}
