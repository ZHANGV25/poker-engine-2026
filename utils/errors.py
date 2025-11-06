"""
Error handling decorators for poker match management.

Provides decorators to handle errors gracefully, determine winners when appropriate,
and log comprehensive error context for debugging.
"""

import functools
import logging
import traceback
from typing import Any, Callable, Optional, Dict


def _extract_error_context(instance: Any) -> Dict[str, Any]:
    """
    Extract error context from a PokerMatch instance.

    Args:
        instance: The PokerMatch instance

    Returns:
        Dictionary of error context fields
    """
    context = {}

    # Common fields that provide useful debugging context
    fields_to_extract = [
        'hand_number',
        'bankrolls',
        'street',
        'acting_agent',
        'time_used',
        'bets',
        'num_hands'
    ]

    for field in fields_to_extract:
        if hasattr(instance, field):
            try:
                context[field] = getattr(instance, field)
            except Exception:
                # Don't fail while extracting error context
                context[field] = "<unable to extract>"

    return context


def _determine_winner_from_error(
    error: Exception,
    instance: Any
) -> Optional[int]:
    """
    Determine which player should win based on the error type.

    Args:
        error: The exception that was raised
        instance: The PokerMatch instance

    Returns:
        Player ID (0 or 1) of the winner, or None if cannot determine
    """
    # Import here to avoid circular imports
    from match import AgentFailure, TimeoutError as MatchTimeoutError

    if isinstance(error, AgentFailure):
        # AgentFailure has failed_player attribute
        if hasattr(error, 'failed_player'):
            failed_player = error.failed_player
            # Winner is the other player
            return 1 - failed_player
        # If both players failed, no winner
        return None

    elif isinstance(error, MatchTimeoutError):
        # TimeoutError occurs when acting player times out
        if hasattr(instance, 'acting_agent'):
            failed_player = instance.acting_agent
            # Winner is the other player
            return 1 - failed_player
        return None

    # For other errors, cannot determine winner
    return None


def _create_error_result(
    instance: Any,
    error: Exception,
    winner: Optional[int] = None
) -> Any:
    """
    Create an error MatchResult.

    Args:
        instance: The PokerMatch instance
        error: The exception that occurred
        winner: Optional winner player ID

    Returns:
        MatchResult object with error information
    """
    # Get current bankrolls or defaults
    bankrolls = [0, 0]
    if hasattr(instance, 'bankrolls'):
        bankrolls = instance.bankrolls

    # Get total hands played
    hands_played = 0
    if hasattr(instance, 'hand_number'):
        hands_played = instance.hand_number

    # Create result dict
    result = {
        'error': True,
        'error_type': type(error).__name__,
        'error_message': str(error),
        'bankrolls': bankrolls,
        'hands_played': hands_played,
        'winner': winner,
        'reason': f"Match terminated due to {type(error).__name__}"
    }

    # If instance has _create_result method, use it
    if hasattr(instance, '_create_result'):
        try:
            return instance._create_result(
                reason=result['reason'],
                winner=winner
            )
        except Exception:
            # If _create_result fails, return dict
            pass

    return result


def handle_game_errors(
    determine_winner: bool = True,
    log_traceback: bool = True,
    reraise: bool = False,
    fallback_action: str = 'return_error_result'
) -> Callable:
    """
    Decorator to handle game errors with mixed behavior based on error type.

    For known errors (AgentFailure, TimeoutError), determines winner and returns appropriate result.
    For unknown errors, logs full traceback and returns error result without winner.

    Args:
        determine_winner: Whether to try to determine a winner from the error
        log_traceback: Whether to log the full stack trace
        reraise: Whether to reraise the exception after handling
        fallback_action: What to do on error - 'return_error_result' or 'reraise'

    Example:
    ```
        @handle_game_errors(determine_winner=True, log_traceback=True)
        def run(self):
            ...
    ```
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get logger and instance
            logger = None
            instance = None
            if args and hasattr(args[0], 'logger'):
                instance = args[0]
                logger = instance.logger
            if logger is None:
                logger = logging.getLogger(func.__module__)

            # Get method name for logging
            method_name = func.__name__
            if instance and hasattr(instance, '__class__'):
                method_name = f"{instance.__class__.__name__}.{func.__name__}"

            try:
                # Execute the method
                return func(*args, **kwargs)

            except Exception as error:
                # Extract error context
                context = {}
                if instance:
                    context = _extract_error_context(instance)

                # Log the error
                error_type = type(error).__name__
                logger.error(
                    f"Error in {method_name}: {error_type}: {str(error)}"
                )

                # Log context
                if context:
                    logger.error(f"Error context: {context}")

                # Log full traceback if requested
                if log_traceback:
                    tb_str = traceback.format_exc()
                    logger.error(f"Traceback:\n{tb_str}")

                # Determine winner if requested and possible
                winner = None
                if determine_winner and instance:
                    winner = _determine_winner_from_error(error, instance)
                    if winner is not None:
                        logger.info(f"Determined winner: Player {winner} (other player caused error)")

                # Decide whether to reraise or return result
                if reraise:
                    raise

                # Return error result
                if instance:
                    result = _create_error_result(instance, error, winner)
                    logger.info(f"Returning error result: {result}")
                    return result
                else:
                    # No instance, reraise
                    raise

        return wrapper
    return decorator


def handle_errors_with_fallback(
    fallback_value: Any = None,
    log_error: bool = True,
    log_traceback: bool = False
) -> Callable:
    """
    Simple error handler that returns a fallback value on error.

    Useful for non-critical methods where you want to continue execution.

    Args:
        fallback_value: Value to return if an error occurs
        log_error: Whether to log the error
        log_traceback: Whether to log the full traceback

    Example:
        @handle_errors_with_fallback(fallback_value=False)
        def _broadcast_to_inactive_players(self, observation, info):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as error:
                if log_error:
                    # Get logger
                    logger = None
                    if args and hasattr(args[0], 'logger'):
                        logger = args[0].logger
                    if logger is None:
                        logger = logging.getLogger(func.__module__)

                    # Get method name
                    method_name = func.__name__
                    if args and hasattr(args[0], '__class__'):
                        method_name = f"{args[0].__class__.__name__}.{func.__name__}"

                    logger.error(f"Error in {method_name}: {type(error).__name__}: {str(error)}")

                    if log_traceback:
                        tb_str = traceback.format_exc()
                        logger.error(f"Traceback:\n{tb_str}")

                return fallback_value

        return wrapper
    return decorator


def log_and_suppress_errors(
    level: int = logging.WARNING
) -> Callable:
    """
    Decorator that logs errors but suppresses them (returns None).

    Useful for cleanup or notification methods where failures shouldn't stop execution.

    Args:
        level: Logging level for the error

    Example:
        @log_and_suppress_errors(level=logging.WARNING)
        def _send_final_observations(self):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as error:
                # Get logger
                logger = None
                if args and hasattr(args[0], 'logger'):
                    logger = args[0].logger
                if logger is None:
                    logger = logging.getLogger(func.__module__)

                # Get method name
                method_name = func.__name__
                if args and hasattr(args[0], '__class__'):
                    method_name = f"{args[0].__class__.__name__}.{func.__name__}"

                logger.log(
                    level,
                    f"Suppressed error in {method_name}: {type(error).__name__}: {str(error)}"
                )
                return None

        return wrapper
    return decorator
