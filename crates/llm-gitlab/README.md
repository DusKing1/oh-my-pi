# omp-llm-gitlab

Protocol edge for GitLab Duo Workflow's authenticated WebSocket agent tunnel.

The crate maps canonical chat history to the workflow service's flat ChatML goal, correlates workflow/session/action identifiers, streams checkpoint text and MCP tool calls, returns executor results on the same socket, and reconnects with an explicit resume request. Dropping the returned stream drops and closes the socket; cancellation is never converted into provider fallback.

Authentication is injected through `WorkflowAuth`. Production callers redeem a canonical broker credential lease into handshake headers, so the transport never accepts or exports a raw token. GitLab's `gitlab-duo-workflow` catalog OAuth row owns the external-redirect PKCE exchange and refresh lifecycle.
