"""Build and publish immutable views of the selected session."""

from alpha_forge.application.events import SessionView, SessionViewChanged
from alpha_forge.application.router import ApplicationEventRouter
from alpha_forge.sessions import Session


def session_view(session: Session) -> SessionView:
    return SessionView(session.session_id, session.revision, session.ui_history())


def publish_session_view(
    event_router: ApplicationEventRouter,
    session: Session,
    *,
    reset_active: bool = False,
) -> None:
    event_router.publish(SessionViewChanged(session_view(session), reset_active))
