import functools
import sys
import traceback


def print_exception_details(exc: BaseException, context: str = "Unhandled exception", tb=None):
    traceback_obj = tb if tb is not None else exc.__traceback__
    print(f"[ERROR] {context}: {type(exc).__name__}: {exc}", file=sys.stderr)
    if traceback_obj is not None:
        frames = traceback.extract_tb(traceback_obj)
        if frames:
            last = frames[-1]
            print(
                f"[ERROR] Location: file={last.filename}, line={last.lineno}, func={last.name}",
                file=sys.stderr,
            )
            if last.line:
                print(f"[ERROR] Code: {last.line.strip()}", file=sys.stderr)
    traceback.print_exception(type(exc), exc, traceback_obj, file=sys.stderr)


def guarded(context: str):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as exc:
                print_exception_details(exc, context=context)
                raise

        return wrapper

    return decorator
