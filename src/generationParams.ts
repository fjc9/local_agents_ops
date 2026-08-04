import type { GenerationParams } from "./types";

export type GenerationParamKey = keyof GenerationParams;

export interface ParamDescriptor {
  key: GenerationParamKey;
  /** Row label, in plain Japanese rather than the engine's term. */
  label: string;
  /** The name Ollama's own documentation uses, kept beside the Japanese one so
   * a user who goes reading upstream docs can match the two up. */
  englishLabel: string;
  /** One line on what moving this actually does to the answers, phrased for
   * someone who has never tuned a sampler. */
  help: string;
  /** What the engine does when this is left unset. Shown as the placeholder
   * and slider position, never sent -- an untouched knob stays absent from the
   * request so the model's own Modelfile value wins. */
  fallback: number;
  min: number;
  max: number;
  step: number;
}

/** The nine knobs, in the order the settings panel lists them: how much the
 * model can read, then how it picks each word, then how it avoids repeating
 * itself. */
export const PARAM_DESCRIPTORS: ParamDescriptor[] = [
  {
    key: "num_ctx",
    label: "覚えていられる長さ",
    englishLabel: "Context Size",
    help: "一度に読み込める会話の量。大きいほど前のやり取りを覚えていますが、メモリ消費と最初の待ち時間が増えます。",
    fallback: 4096,
    min: 512,
    max: 32768,
    step: 512,
  },
  {
    key: "temperature",
    label: "回答のばらつき",
    englishLabel: "Temperature",
    help: "低いほど毎回ほぼ同じ堅実な答えになり、高いほど表現が多様で意外な言い回しも出ます。",
    fallback: 0.8,
    min: 0,
    max: 2,
    step: 0.05,
  },
  {
    key: "top_k",
    label: "候補に残す単語の数",
    englishLabel: "Top K",
    help: "次の1語の候補を上位いくつまでに絞るか。小さいほど無難、大きいほど自由。0で絞り込みなし。",
    fallback: 40,
    min: 0,
    max: 200,
    step: 1,
  },
  {
    key: "top_p",
    label: "候補に残す確率の幅",
    englishLabel: "Top P",
    help: "確率の高い候補から合計何割ぶんまでを残すか。小さいほど堅実、大きいほど大胆になります。",
    fallback: 0.9,
    min: 0,
    max: 1,
    step: 0.05,
  },
  {
    key: "min_p",
    label: "候補を切り捨てる下限",
    englishLabel: "Min P",
    help: "最有力の候補に比べて確率がこの割合未満の単語を捨てます。上げるほど的外れな語が減ります。",
    fallback: 0,
    min: 0,
    max: 1,
    step: 0.01,
  },
  {
    key: "repeat_last_n",
    label: "繰り返しを見張る範囲",
    englishLabel: "Repeat Last N",
    help: "直前いくつぶんの単語までさかのぼって重複を監視するか。0で監視しない、-1で会話全体。",
    fallback: 64,
    min: -1,
    max: 2048,
    step: 1,
  },
  {
    key: "repeat_penalty",
    label: "繰り返しを抑える強さ",
    englishLabel: "Repeat Penalty",
    help: "同じ表現が再登場するのをどれだけ嫌うか。上げると言い換えますが、上げすぎると不自然になります。",
    fallback: 1.1,
    min: 0,
    max: 2,
    step: 0.05,
  },
  {
    key: "presence_penalty",
    label: "同じ話題から離れる強さ",
    englishLabel: "Presence Penalty",
    help: "一度出た単語を回数に関わらず一律に減点し、新しい話題へ移りやすくします。",
    fallback: 0,
    min: -2,
    max: 2,
    step: 0.1,
  },
  {
    key: "frequency_penalty",
    label: "多用する語を抑える強さ",
    englishLabel: "Frequency Penalty",
    help: "何度も出てきた単語ほど強く減点します。口癖のような繰り返しを抑えられます。",
    fallback: 0,
    min: -2,
    max: 2,
    step: 0.1,
  },
];

/** Decimal places to show, derived from the step so 0.05 reads as "0.85" and a
 * whole-number knob never picks up a trailing ".0". */
export function formatParamValue(value: number, step: number): string {
  if (Number.isInteger(step)) return String(Math.round(value));
  const decimals = (String(step).split(".")[1] ?? "").length;
  return value.toFixed(decimals);
}

/** Whether the user has set anything at all for this model, i.e. whether
 * there's an override to offer to clear. */
export function hasOverrides(params: GenerationParams | undefined): boolean {
  return !!params && PARAM_DESCRIPTORS.some((d) => params[d.key] != null);
}
