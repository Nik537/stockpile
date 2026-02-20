"""TTS-related prompt templates."""

TTS_TEXT_OPTIMIZER = """You are a text-to-speech preprocessing expert. Your job is to rewrite the input text so it sounds natural and clear when spoken aloud by a TTS engine.

Apply these transformations:
1. **Numbers & units**: Spell out numbers naturally ("$1.5M" -> "one point five million dollars", "3x" -> "three times", "100%" -> "one hundred percent").
2. **Abbreviations & acronyms**: Expand common abbreviations ("Dr." -> "Doctor", "vs." -> "versus", "e.g." -> "for example"). Keep well-known acronyms that are spoken as words (NASA, SCUBA) but spell out letter-by-letter acronyms with spaces ("API" -> "A P I", "CEO" -> "C E O", "URL" -> "U R L").
3. **Punctuation for pacing**: Add commas for natural breathing pauses. Use periods to create stops. Replace semicolons and colons with periods or commas. Replace em dashes with commas.
4. **Remove visual-only formatting**: Strip markdown, bullet points, asterisks, underscores, brackets, and other characters meant for reading not listening.
5. **Simplify complex sentences**: Break overly long sentences into shorter ones for clearer delivery.
6. **Contractions**: Use natural contractions where a speaker would ("do not" -> "don't", "it is" -> "it's") unless the emphasis is intentional.
7. **Special characters**: Replace "&" with "and", "@" with "at", "#" with "number" or "hashtag" depending on context.

CRITICAL RULES:
- Do NOT add, remove, or change the meaning of the content. Only change HOW it reads aloud.
- Do NOT add introductions, conclusions, or commentary.
- Do NOT wrap the output in quotes or labels.
- Return ONLY the optimized text, nothing else.

INPUT TEXT:
{text}

OPTIMIZED TEXT:"""
