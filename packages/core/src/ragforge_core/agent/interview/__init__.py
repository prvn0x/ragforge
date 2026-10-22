"""Interview mode (Phases 3b/3c, docs/NEXT.md "TO DO NEXT" 1): one grounded question at a time,
graded against the notes with a verified model answer, at a chosen level (SDE-1 .. Staff), or,
when the client explicitly asks for it, one general-knowledge question graded by the model alone
(never cited, never verified). Stateless: the session lives in the client, which sends back the
chunk ids (or, in general mode, the questions) already asked.

Layout (each decision step is a pure function an offline eval can call with a fake or real chat
model and fixed notes, without HTTP or a database):
- `selection`: which note to ask about, and which kind of question (pure);
- `acronyms`: "HLD" -> "High Level Design" over the visible titles and headings (pure);
- `prompts`: the (system, user) pair of each call, untrusted text fenced as data (pure);
- `replies`: strict JSON parsing, the one repair retry, the per-call log line;
- `verify`: the chat path's verification, applied to the model answer and grading items;
- `question` / `grade`: one question / one grade from given notes or from general knowledge
  (model calls only);
- `service`: the same two operations with their reads (tenant transactions, retrieval);
- `tokens`: the HMAC `question_token` binding a question to its principal;
- `ports`: the SQL reads the service needs (implemented in the API layer).
"""
