import os

import streamlit as st

from magic_assistant.agent.rendering import CUSTOM_CARD_NOTICE
from magic_assistant.api.schemas import ChatResponse
from magic_assistant.ui.client import (
    DemoApiClient,
    DemoApiError,
    new_thread_id,
    response_presentation,
    source_label,
    technical_details,
    valid_image_url,
)

st.set_page_config(page_title="Magic Assistant", page_icon="M", layout="centered")
st.title("Magic: The Gathering Assistant")
st.caption("Grounded rules answers, card search, interactions, and fan-made card concepts.")


@st.cache_resource
def api_client() -> DemoApiClient:
    return DemoApiClient(os.environ.get("MAGIC_API_URL", "http://localhost:8000"))


if "thread_id" not in st.session_state:
    st.session_state.thread_id = new_thread_id()
if "messages" not in st.session_state:
    st.session_state.messages = []

with st.sidebar:
    st.subheader("Demo controls")
    if st.button("New conversation", use_container_width=True):
        st.session_state.thread_id = new_thread_id()
        st.session_state.messages = []
        st.rerun()
    if api_client().ready():
        st.success("Backend ready")
    else:
        st.error("Backend unavailable")
    st.caption(f"Thread: {st.session_state.thread_id}")


def render_structured_custom_card(response: ChatResponse) -> None:
    card = response.custom_card
    if card is None:
        return
    st.warning(CUSTOM_CARD_NOTICE)
    st.subheader(card.name)
    if card.mana_cost:
        st.write(f"Mana cost: {card.mana_cost}")
    if card.colors:
        st.write(f"Colors: {''.join(color.value for color in card.colors)}")
    st.write(f"Type: {card.type_line}")
    st.write(card.oracle_text)
    if card.power is not None:
        st.write(f"Power: {card.power}")
    if card.toughness is not None:
        st.write(f"Toughness: {card.toughness}")
    if card.flavor_text:
        st.caption(card.flavor_text)


def render_response(response: ChatResponse) -> None:
    presentation = response_presentation(response)
    if presentation.show_custom_card:
        render_structured_custom_card(response)
    if presentation.answer_text:
        st.markdown(presentation.answer_text)
    if presentation.card_heading:
        st.markdown(f"**{presentation.card_heading}**")
    for card in response.cards:
        st.subheader(card.name)
        st.caption(
            f"{card.mana_cost or 'No mana cost'} · mana value {card.mana_value} · "
            f"{card.type_line or 'Unknown type'}"
        )
        if card.oracle_text:
            st.write(card.oracle_text)
        if valid_image_url(card.image_url):
            st.image(card.image_url, width=300)
    if response.sources:
        st.markdown("**Sources**")
        for source in response.sources:
            st.write(f"- {source_label(source)}")
    with st.expander("Technical details"):
        details = technical_details(response)
        st.write(f"Intent: `{details['intent']}`")
        st.write(f"Graph route: {details['route']}")
        st.write(f"Request ID: `{details['request_id']}`")
        if details["sources"]:
            st.write("Sources:")
            for source in details["sources"]:
                st.write(f"- {source}")


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "assistant" and message.get("response"):
            render_response(ChatResponse.model_validate(message["response"]))
        else:
            st.markdown(message["content"])

if prompt := st.chat_input("Ask about Magic rules, cards, or interactions"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Consulting rules and card data..."):
                response = api_client().chat(prompt, st.session_state.thread_id)
            render_response(response)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": response.answer,
                    "response": response.model_dump(mode="json"),
                }
            )
        except DemoApiError:
            message = "The backend could not complete the request. Check API readiness and logs."
            st.error(message)
            st.session_state.messages.append(
                {"role": "assistant", "content": message, "response": None}
            )
