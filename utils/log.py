"""
Logging decorators for poker match tracking.

Provides decorators to automatically log method calls, game state changes,
and player actions with configurable verbosity and formatting.
"""

import functools
import logging
import time
import json
from typing import Any, Callable, List, Optional, Dict


def _format_value(value: Any, max_length: int = 100) -> str:
    """Format a value for logging, truncating if necessary."""
    if isinstance(value, (dict, list)):
        try:
            formatted = json.dumps(value, indent=2)
            if len(formatted) > max_length:
                return formatted[:max_length] + "..."
            return formatted
        except (TypeError, ValueError):
            str_val = str(value)
            return str_val[:max_length] + "..." if len(str_val) > max_length else str_val
    else:
        str_val = str(value)
        return str_val[:max_length] + "..." if len(str_val) > max_length else str_val


def _extract_game_state(instance: Any, fields: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Extract game state from a PokerMatch instance.

    Args:
        instance: The PokerMatch instance
        fields: Specific fields to extract. If None, extracts common fields.

    Returns:
        Dictionary of extracted state fields
    """
    if fields is None:
        # Default fields to extract
        fields = ['hand_number', 'bankrolls', 'street', 'acting_agent', 'time_used']

    state = {}
    for field in fields:
        if hasattr(instance, field):
            value = getattr(instance, field)
            state[field] = value

    return state


def log_method(
    level: int = logging.DEBUG,
    log_entry: bool = True,
    log_exit: bool = True,
    log_duration: bool = True,
    include_args: bool = False,
    include_result: bool = False,
    max_result_length: int = 200
) -> Callable:
    """
    Decorator to log method entry, exit, and execution time.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_entry: Whether to log when method is entered
        log_exit: Whether to log when method exits
        log_duration: Whether to log execution duration
        include_args: Whether to include method arguments in entry log
        include_result: Whether to include return value in exit log
        max_result_length: Maximum length for logged result

    Example:
        @log_method(level=logging.INFO, log_duration=True)
        def my_method(self, arg1):
            return "result"
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get logger - try to get from instance (self), otherwise use module logger
            logger = None
            if args and hasattr(args[0], 'logger'):
                logger = args[0].logger
            if logger is None:
                logger = logging.getLogger(func.__module__)

            # Get class name if this is a method
            class_name = ""
            if args and hasattr(args[0], '__class__'):
                class_name = f"{args[0].__class__.__name__}."

            method_name = f"{class_name}{func.__name__}"

            # Log entry
            if log_entry:
                if include_args and (args[1:] or kwargs):
                    args_str = ", ".join([_format_value(arg, 50) for arg in args[1:]])
                    kwargs_str = ", ".join([f"{k}={_format_value(v, 50)}" for k, v in kwargs.items()])
                    params = ", ".join(filter(None, [args_str, kwargs_str]))
                    logger.log(level, f"→ {method_name}({params})")
                else:
                    logger.log(level, f"→ {method_name}")

            # Execute method and track time
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time

                # Log exit
                if log_exit:
                    exit_parts = [f"← {method_name}"]
                    if log_duration:
                        exit_parts.append(f"[{duration:.3f}s]")
                    if include_result and result is not None:
                        exit_parts.append(f"→ {_format_value(result, max_result_length)}")
                    logger.log(level, " ".join(exit_parts))

                return result

            except Exception as e:
                duration = time.time() - start_time
                if log_exit:
                    logger.log(level, f"✗ {method_name} [{duration:.3f}s] - raised {type(e).__name__}")
                raise

        return wrapper
    return decorator


def log_game_state(
    level: int = logging.DEBUG,
    fields: Optional[List[str]] = None,
    on_entry: bool = False,
    on_exit: bool = True,
    format_json: bool = True
) -> Callable:
    """
    Decorator to log game state from a PokerMatch instance.

    Automatically extracts specified fields from self and logs them.

    Args:
        level: Logging level
        fields: List of field names to extract from self.
                If None, uses defaults: ['hand_number', 'bankrolls', 'street', 'acting_agent', 'time_used']
        on_entry: Whether to log state when entering method
        on_exit: Whether to log state when exiting method
        format_json: Whether to format state as JSON

    Example:
        @log_game_state(fields=['hand_number', 'bankrolls'], on_exit=True)
        def _play_hand(self):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Get logger from instance
            logger = None
            instance = None
            if args and hasattr(args[0], 'logger'):
                instance = args[0]
                logger = instance.logger
            if logger is None:
                logger = logging.getLogger(func.__module__)

            # Get method name
            class_name = ""
            if instance and hasattr(instance, '__class__'):
                class_name = f"{instance.__class__.__name__}."
            method_name = f"{class_name}{func.__name__}"

            # Log state on entry
            if on_entry and instance:
                state = _extract_game_state(instance, fields)
                if format_json:
                    state_str = json.dumps(state, indent=2)
                    logger.log(level, f"[ENTRY] {method_name} - Game State:\n{state_str}")
                else:
                    logger.log(level, f"[ENTRY] {method_name} - {state}")

            # Execute method
            result = func(*args, **kwargs)

            # Log state on exit
            if on_exit and instance:
                state = _extract_game_state(instance, fields)
                if format_json:
                    state_str = json.dumps(state, indent=2)
                    logger.log(level, f"[EXIT] {method_name} - Game State:\n{state_str}")
                else:
                    logger.log(level, f"[EXIT] {method_name} - {state}")

            return result

        return wrapper
    return decorator


def log_action(
    level: int = logging.INFO,
    include_timing: bool = True,
    include_game_context: bool = True
) -> Callable:
    """
    Decorator specialized for logging player actions in poker match.

    Automatically extracts action details and game context for comprehensive action logging.

    Args:
        level: Logging level
        include_timing: Whether to include time taken for action
        include_game_context: Whether to include hand_number, street, etc.

    Example:
        @log_action(level=logging.INFO, include_timing=True)
        def _get_action_from_active_player(self):
            ...
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

            # Extract game context before action
            context = {}
            if include_game_context and instance:
                context = _extract_game_state(instance, ['hand_number', 'street', 'acting_agent'])

            # Execute method and track time
            start_time = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - start_time

            # Build log message
            log_parts = []

            if include_game_context and context:
                log_parts.append(f"Hand {context.get('hand_number', '?')}")
                log_parts.append(f"Street {context.get('street', '?')}")
                log_parts.append(f"Player {context.get('acting_agent', '?')}")

            # Try to extract action info from result if it's a tuple/dict
            action_info = "action completed"
            if isinstance(result, tuple) and len(result) >= 1:
                action_info = f"action={result[0]}"
                if len(result) >= 2:
                    action_info += f", amount={result[1]}"
            elif isinstance(result, dict):
                action_info = f"action={result.get('action', '?')}"
                if 'amount' in result:
                    action_info += f", amount={result['amount']}"

            log_parts.append(action_info)

            if include_timing:
                log_parts.append(f"[{duration:.3f}s]")

            logger.log(level, " | ".join(log_parts))

            return result

        return wrapper
    return decorator


def log_state_change(
    level: int = logging.DEBUG,
    message_template: Optional[str] = None,
    fields: Optional[List[str]] = None
) -> Callable:
    """
    Decorator to log specific state changes with a custom message.

    Args:
        level: Logging level
        message_template: Template string for log message. Can use {field_name} placeholders.
        fields: Fields to extract from self for the template

    Example:
        @log_state_change(
            message_template="Bankrolls updated: Player 0=${bankrolls[0]}, Player 1=${bankrolls[1]}",
            fields=['bankrolls']
        )
        def _update_bankrolls(self, pot):
            ...
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Execute method first
            result = func(*args, **kwargs)

            # Get logger and instance
            logger = None
            instance = None
            if args and hasattr(args[0], 'logger'):
                instance = args[0]
                logger = instance.logger
            if logger is None:
                logger = logging.getLogger(func.__module__)

            # Log state change
            if instance:
                if message_template and fields:
                    state = _extract_game_state(instance, fields)
                    try:
                        message = message_template.format(**state)
                        logger.log(level, message)
                    except (KeyError, IndexError):
                        logger.log(level, f"State change in {func.__name__}: {state}")
                else:
                    state = _extract_game_state(instance, fields)
                    logger.log(level, f"State change in {func.__name__}: {state}")

            return result

        return wrapper
    return decorator
