interface ErrorResponseBody {
  error?: unknown;
  message?: unknown;
  request_id?: unknown;
}

export class ApiError extends Error {
  readonly status: number;
  readonly code: string | null;
  readonly requestId: string | null;

  constructor(
    message: string,
    status: number,
    code: string | null,
    requestId: string | null,
  ) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.requestId = requestId;
  }
}

function stringField(value: unknown): string | null {
  return typeof value === "string" ? value : null;
}

export async function requestJson<T>(
  input: RequestInfo | URL,
  init?: RequestInit,
): Promise<T> {
  const response = await fetch(input, init);
  const body: unknown = await response.json().catch(() => null);

  if (!response.ok) {
    const errorBody =
      body !== null && typeof body === "object"
        ? (body as ErrorResponseBody)
        : null;
    throw new ApiError(
      stringField(errorBody?.message) ??
        `La API respondió con estado ${response.status}.`,
      response.status,
      stringField(errorBody?.error),
      stringField(errorBody?.request_id),
    );
  }

  return body as T;
}
