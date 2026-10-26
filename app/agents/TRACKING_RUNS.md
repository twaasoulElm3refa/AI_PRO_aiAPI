# Tracking agent runs in the UI

The synchronous run endpoint supports live progress polling when the client sends
its own UUID in `run_uuid`.

## Recommended flow

1. Generate a UUID in the client.
2. Start `POST /tasks/agents/{agent_slug}/run` with that `run_uuid`.
3. Do not wait for the POST promise before starting the progress indicator.
4. Poll `GET /tasks/agents/runs/{run_uuid}/status?user_id={user_id}` every 1-2
   seconds.
5. Stop polling when `progress.is_terminal` is `true`, then load `result_url`.

The browser should call your application backend (for example Laravel), which
proxies these requests to FastAPI and adds `X-Internal-Api-Key` server-side.
Never expose the internal API key in browser JavaScript.

Example browser flow against application-owned proxy routes:

```javascript
const runUuid = crypto.randomUUID();
const payload = { ...agentRequest, run_uuid: runUuid };

const resultPromise = fetch(`/api/agents/${agentSlug}/run`, {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(payload),
}).then((response) => response.json());

const timer = setInterval(async () => {
  const response = await fetch(
    `/api/agents/runs/${runUuid}/status?user_id=${payload.user_id}`,
  );
  if (response.status === 404) return; // The run is still being created.

  const data = await response.json();
  renderProgress(data.progress.label_ar, data.progress.percent);
  if (data.progress.is_terminal) clearInterval(timer);
}, 1500);

const finalResult = await resultPromise;
clearInterval(timer);
```

The progress phases are `start`, `planner`, `validate_plan`, `execute_tools`,
`generator`, `evaluator`, `revision`, and `finalize`. Display `label_ar` or
`label`, and use `percent` only as a phase indicator rather than an exact time
estimate. A revision can move the displayed percentage backward because the
agent is improving an answer after quality review.
