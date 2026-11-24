#!/usr/bin/env python3
"""Outbound Confirmation Agent — Proactive appointment confirmation calls.

A stripped-down agent for outbound calls to confirm, reschedule, or cancel
upcoming appointments, and to remind patients about recall appointments.
Simpler state machine than the inbound DentalAgent.
"""

import logging
import json
from datetime import datetime, timedelta
from signalwire import AgentBase
from signalwire.core.function_result import SwaigFunctionResult

import config
import opendental_client as od
from state_store import load_call_state, save_call_state, save_call_log

logger = logging.getLogger(__name__)


class OutboundConfirmAgent(AgentBase):
    """Outbound agent for appointment confirmations and recall reminders."""

    def __init__(self):
        super().__init__(name="OutboundConfirmAgent", route="/outbound",
                         record_call=True, record_format="wav", record_stereo=True)

        # AI model
        self.set_param("ai_model", config.AI_MODEL)
        self.set_param("wait_for_user", True)
        self.set_prompt_llm_params(top_p=config.AI_TOP_P, temperature=config.AI_TEMPERATURE)

        # Personality
        self.prompt_add_section("Personality",
            f"You are a friendly, professional outbound caller from {config.PRACTICE_NAME}. "
            "You are calling patients to confirm upcoming appointments or remind them about "
            "recall/cleaning appointments. Keep it brief and warm — this is a quick courtesy call."
        )

        # Voice behavior
        self.prompt_add_section("Rules", body="", bullets=[
            "This is a PHONE CALL. Keep every response to 1-2 short sentences.",
            "Say times naturally: 'nine thirty AM' not '09:30'. Say dates naturally: 'Tuesday February 26th' not '2026-02-26'.",
            "Use provider titles: 'Dr. Smith' not 'provider 1'. Use first names for hygienists.",
            "Be respectful of the patient's time — this is an outbound call they didn't initiate.",
            "If you reach voicemail, leave a brief message and end the call.",
        ])

        # Voice
        self.add_language("English", "en-US", "azure.en-US-AvaNeural")
        self.add_hints(["confirm", "reschedule", "cancel", "appointment",
                        "cleaning", "checkup", "yes", "no"])

        self.set_post_prompt("Summarize the outbound call outcome.")

        # Cache providers
        self._providers_cache = {}
        self._load_providers_cache()

        # State machine
        self._define_state_machine()

        # SWAIG tools
        self._define_tools()

        # Per-call config — outbound calls inject appointment/recall data
        self.set_dynamic_config_callback(self._per_call_config)

    def _load_providers_cache(self):
        """Pre-load provider names."""
        try:
            providers = od.get_providers()
            if providers and isinstance(providers, list):
                for p in providers:
                    prov_num = p.get("ProvNum")
                    if prov_num and not p.get("IsHidden"):
                        lname = p.get("LName", "")
                        abbr = p.get("Abbr", "")
                        if lname:
                            self._providers_cache[prov_num] = f"Dr. {lname}"
                        elif abbr:
                            self._providers_cache[prov_num] = abbr
        except Exception as e:
            logger.warning(f"Could not pre-load providers: {e}")

    # ── State Machine ────────────────────────────────────────────────

    def _define_state_machine(self):
        """Define outbound conversation flow.

        Step order matters for gather next_step transitions.
        """
        contexts = self.define_contexts()
        ctx = contexts.add_context("default")

        # GREETING — customized per-call by _per_call_config
        greeting = ctx.add_step("greeting")
        greeting.add_section("Task", "Introduce yourself and the reason for the call")
        greeting.set_functions("none")
        greeting.set_valid_steps(["confirm_or_act", "offer_scheduling"])

        # ── Appointment Confirmation Flow ────────────────────────────

        confirm_or_act = ctx.add_step("confirm_or_act")
        confirm_or_act.add_section("Task", "Present the appointment and ask for confirmation")
        confirm_or_act.add_bullets("Process", [
            "Read the appointment details from the outbound context",
            "Ask: 'Can you confirm you'll be there?'",
            "If yes, call confirm_outbound",
            "If they need to reschedule, proceed to handle_reschedule",
            "If they want to cancel, proceed to handle_cancel",
        ])
        confirm_or_act.set_step_criteria("Patient confirms, reschedules, or cancels")
        confirm_or_act.set_functions(["confirm_outbound"])
        confirm_or_act.set_valid_steps(["handle_reschedule", "handle_cancel", "log_and_end"])

        handle_cancel = ctx.add_step("handle_cancel")
        handle_cancel.add_section("Task", "Confirm the patient wants to cancel")
        handle_cancel.add_bullets("Process", [
            "Ask: 'Are you sure you'd like to cancel your appointment?'",
            "If yes, call cancel_outbound",
            "If they change their mind, call confirm_outbound instead",
        ])
        handle_cancel.set_step_criteria("Cancel confirmed or reversed")
        handle_cancel.set_functions(["cancel_outbound", "confirm_outbound"])
        handle_cancel.set_valid_steps(["log_and_end"])

        # collect_resched_prefs (gather) — next_step → search_resched_slots
        ctx.add_step("handle_reschedule") \
            .set_text("Collect new preferences for rescheduling.") \
            .set_functions("none") \
            .set_gather_info(
                output_key="resched_prefs_answers",
                completion_action="next_step",
                prompt="Let's find a new time that works better for you."
            ) \
            .add_gather_question(
                "new_preferred_date",
                "What date would work better for you?",
                confirm=True,
                prompt="Accept natural language but submit in YYYY-MM-DD format. Must be a future date."
            ) \
            .add_gather_question(
                "time_preference",
                "Morning or afternoon?",
                prompt="Submit exactly: morning, afternoon, or no_preference"
            ) \
            .set_valid_steps(["search_resched_slots"])

        # search_resched_slots — MUST be right after handle_reschedule (next_step target)
        search_resched_slots = ctx.add_step("search_resched_slots")
        search_resched_slots.add_section("Task", "Search for new appointment slots")
        search_resched_slots.add_bullets("Process", [
            "Call search_available_slots to find openings",
        ])
        search_resched_slots.set_step_criteria("Slots found")
        search_resched_slots.set_functions(["search_available_slots"])
        search_resched_slots.set_valid_steps(["present_resched_options", "log_and_end"])

        present_resched_options = ctx.add_step("present_resched_options")
        present_resched_options.add_section("Task", "Present new time slots")
        present_resched_options.add_bullets("Process", [
            "Read the available slots",
            "When the patient picks one, call reschedule_outbound with that option_number",
            "If none work, apologize and say we'll call back with more options",
        ])
        present_resched_options.set_step_criteria("Patient picks a slot or declines")
        present_resched_options.set_functions(["reschedule_outbound"])
        present_resched_options.set_valid_steps(["log_and_end", "handle_reschedule"])

        # ── Recall Reminder Flow ─────────────────────────────────────

        offer_scheduling = ctx.add_step("offer_scheduling")
        offer_scheduling.add_section("Task", "Offer to schedule a recall appointment")
        offer_scheduling.add_bullets("Process", [
            "Read the recall details from the outbound context",
            "Ask: 'Would you like to schedule that now?'",
            "If yes, proceed to collect recall preferences",
            "If no, call decline_recall",
        ])
        offer_scheduling.set_step_criteria("Patient accepts or declines scheduling")
        offer_scheduling.set_functions(["decline_recall"])
        offer_scheduling.set_valid_steps(["collect_recall_prefs", "log_and_end"])

        # collect_recall_prefs (gather) — next_step → search_recall_slots
        ctx.add_step("collect_recall_prefs") \
            .set_text("Collect preferences for recall appointment.") \
            .set_functions("none") \
            .set_gather_info(
                output_key="recall_prefs_answers",
                completion_action="next_step",
                prompt="Let's find a good time for your appointment."
            ) \
            .add_gather_question(
                "preferred_date",
                "What date works best for you?",
                confirm=True,
                prompt="Accept natural language but submit in YYYY-MM-DD format. Must be a future date."
            ) \
            .add_gather_question(
                "time_preference",
                "Morning or afternoon?",
                prompt="Submit exactly: morning, afternoon, or no_preference"
            ) \
            .set_valid_steps(["search_recall_slots"])

        # search_recall_slots — MUST be right after collect_recall_prefs (next_step target)
        search_recall_slots = ctx.add_step("search_recall_slots")
        search_recall_slots.add_section("Task", "Search for recall appointment slots")
        search_recall_slots.add_bullets("Process", [
            "Call search_available_slots to find openings for the recall",
        ])
        search_recall_slots.set_step_criteria("Slots found")
        search_recall_slots.set_functions(["search_available_slots"])
        search_recall_slots.set_valid_steps(["present_recall_options", "log_and_end"])

        present_recall_options = ctx.add_step("present_recall_options")
        present_recall_options.add_section("Task", "Present recall appointment options")
        present_recall_options.add_bullets("Process", [
            "Read the available slots",
            "When the patient picks one, call book_recall with that option_number",
            "If none work, say 'No problem, give us a call when you're ready to schedule.'",
        ])
        present_recall_options.set_step_criteria("Patient books or declines")
        present_recall_options.set_functions(["book_recall"])
        present_recall_options.set_valid_steps(["log_and_end", "collect_recall_prefs"])

        # ── Unable to Reach / End ────────────────────────────────────

        unable_to_reach_step = ctx.add_step("unable_to_reach")
        unable_to_reach_step.add_section("Task", "Handle voicemail or no answer")
        unable_to_reach_step.add_bullets("Process", [
            "If voicemail: leave a brief message asking them to call back",
            f"Say: 'Hi, this is {config.PRACTICE_NAME} calling about your upcoming appointment. "
            f"Please give us a call back at your convenience.'",
            "Call unable_to_reach_tool to log the attempt",
        ])
        unable_to_reach_step.set_functions(["unable_to_reach_tool"])
        unable_to_reach_step.set_valid_steps(["log_and_end"])

        log_and_end = ctx.add_step("log_and_end")
        log_and_end.add_section("Task", "Log the call and end gracefully")
        log_and_end.add_bullets("Process", [
            "Call log_outbound_call with the outcome and any notes",
            f"Say a brief goodbye: 'Thanks, have a great day!'",
        ])
        log_and_end.set_functions(["log_outbound_call"])
        log_and_end.set_valid_steps([])

    # ── Per-Call Config ───────────────────────────────────────────────

    def _per_call_config(self, query_params, body_params, headers, agent):
        """Inject appointment/recall data into the outbound call context.

        The confirmation_runner passes data via query parameters:
        - call_type: "confirmation" or "recall"
        - pat_num: Patient number
        - apt_num: Appointment number (for confirmations)
        - recall_type: Recall type (for recalls)
        """
        params = query_params or {}
        call_type = params.get("call_type", "confirmation")
        pat_num = params.get("pat_num")
        apt_num = params.get("apt_num")
        recall_type = params.get("recall_type", "Prophy")

        if not pat_num:
            return

        pat_num = int(pat_num)
        patient = od.get_patient(pat_num)
        if not patient:
            return

        first_name = patient.get("FName", "")
        last_name = patient.get("LName", "")

        ctx = agent._contexts_builder.get_context("default")
        greeting_step = ctx.get_step("greeting")
        greeting_step.clear_sections()
        greeting_step.set_functions("none")

        if call_type == "confirmation" and apt_num:
            apt_num = int(apt_num)
            appointment = od.get_appointment(apt_num)
            if not appointment:
                return

            apt_dt = appointment.get("AptDateTime", "")
            prov_num = appointment.get("ProvNum")
            provider_name = self._providers_cache.get(prov_num, "your provider")
            time_display = od.format_datetime_voice(apt_dt)
            note = appointment.get("Note", "")

            agent.update_global_data({
                "call_type": "confirmation",
                "patient_profile": {
                    "PatNum": pat_num,
                    "first_name": first_name,
                    "last_name": last_name,
                },
                "appointment": {
                    "AptNum": apt_num,
                    "datetime": apt_dt,
                    "ProvNum": prov_num,
                    "provider": provider_name,
                    "time_display": time_display,
                    "note": note,
                },
            })

            greeting_step.set_valid_steps(["confirm_or_act"])
            greeting_step.add_section("Task", "Greet patient and state purpose")
            greeting_step.add_bullets("Process", [
                f"Say: 'Hi {first_name}, this is {config.PRACTICE_NAME} calling about your upcoming appointment.'",
                f"Say: 'You have a {note or 'dental'} appointment on {time_display} with {provider_name}.'",
                "Ask: 'Can you confirm you'll be there?'",
                "Proceed to confirm_or_act",
            ])

            agent.prompt_add_section("Appointment Details",
                f"Patient: {first_name} {last_name}\n"
                f"Appointment: {time_display} with {provider_name}\n"
                f"Procedure: {note or 'General'}\n"
                f"AptNum: {apt_num}"
            )

        elif call_type == "recall":
            agent.update_global_data({
                "call_type": "recall",
                "patient_profile": {
                    "PatNum": pat_num,
                    "first_name": first_name,
                    "last_name": last_name,
                },
                "recall_info": {
                    "recall_type": recall_type,
                },
            })

            recall_label = "cleaning" if recall_type == "Prophy" else recall_type.lower()

            greeting_step.set_valid_steps(["offer_scheduling"])
            greeting_step.add_section("Task", "Greet patient and mention recall")
            greeting_step.add_bullets("Process", [
                f"Say: 'Hi {first_name}, this is {config.PRACTICE_NAME}.'",
                f"Say: 'We wanted to let you know you're due for your {recall_label}.'",
                "Ask: 'Would you like to schedule that now?'",
                "Proceed to offer_scheduling",
            ])

            agent.prompt_add_section("Recall Details",
                f"Patient: {first_name} {last_name}\n"
                f"Recall type: {recall_label}\n"
                f"PatNum: {pat_num}"
            )

    # ── SWAIG Tools ──────────────────────────────────────────────────

    def _define_tools(self):
        """Define outbound agent SWAIG tools."""

        def _call_id(raw_data):
            if not isinstance(raw_data, dict):
                return "unknown"
            return raw_data.get("call_id", "unknown")

        def _change_step(result, step):
            logger.info(f"outbound step_change: -> {step}")
            result.swml_change_step(step)

        providers_cache = self._providers_cache

        # 1. CONFIRM OUTBOUND
        @self.tool(
            name="confirm_outbound",
            description="Confirm the patient's appointment.",
            wait_file="/sounds/typing.mp3",
            function_fillers={"en-US": [
                "Marking your appointment as confirmed.",
                "Updating your confirmation status now.",
                "Locking that in for you.",
            ]},
            parameters={"type": "object", "properties": {}, "required": []},
        )
        def confirm_outbound(args, raw_data):
            global_data = (raw_data or {}).get("global_data", {})
            appointment = global_data.get("appointment", {})
            apt_num = appointment.get("AptNum")
            call_id = _call_id(raw_data)
            state = load_call_state(call_id)

            if apt_num:
                od.confirm_appointment(apt_num, confirm_val="Confirmed")
                today = datetime.now().strftime("%m/%d")
                od.append_appointment_note(apt_num, f"Confirmed via AI call {today}")

            state["outcome"] = "confirmed"
            save_call_state(call_id, state)

            result = SwaigFunctionResult(
                "Your appointment is confirmed! We'll see you then."
            )
            _change_step(result, "log_and_end")
            return result

        # 2. CANCEL OUTBOUND
        @self.tool(
            name="cancel_outbound",
            description="Cancel the patient's appointment after confirmation.",
            wait_file="/sounds/typing.mp3",
            function_fillers={"en-US": [
                "Processing your cancellation now.",
                "Removing that appointment from the schedule.",
            ]},
            parameters={"type": "object", "properties": {}, "required": []},
        )
        def cancel_outbound(args, raw_data):
            global_data = (raw_data or {}).get("global_data", {})
            appointment = global_data.get("appointment", {})
            apt_num = appointment.get("AptNum")
            call_id = _call_id(raw_data)
            state = load_call_state(call_id)

            if apt_num:
                od.cancel_appointment(apt_num)
                today = datetime.now().strftime("%m/%d")
                od.append_appointment_note(apt_num, f"Cancelled via AI call {today}")

            state["outcome"] = "cancelled"
            save_call_state(call_id, state)

            result = SwaigFunctionResult(
                "Your appointment has been cancelled. If you need to rebook, just give us a call."
            )
            _change_step(result, "log_and_end")
            return result

        # 3. SEARCH AVAILABLE SLOTS (reused for reschedule + recall)
        @self.tool(
            name="search_available_slots",
            description="Search for available appointment openings.",
            wait_file="/sounds/typing.mp3",
            function_fillers={"en-US": [
                "Checking the schedule for openings.",
                "Looking for available times.",
                "Searching our appointment book.",
            ]},
            parameters={"type": "object", "properties": {}, "required": []},
        )
        def search_available_slots(args, raw_data):
            global_data = (raw_data or {}).get("global_data", {})
            call_id = _call_id(raw_data)
            state = load_call_state(call_id)

            prefs = (global_data.get("resched_prefs_answers")
                     or global_data.get("recall_prefs_answers") or {})
            preferred_date = prefs.get("preferred_date") or prefs.get("new_preferred_date", "")
            time_pref = prefs.get("time_preference", "no_preference")

            try:
                start_date = datetime.strptime(preferred_date, "%Y-%m-%d")
            except (ValueError, TypeError):
                start_date = datetime.now() + timedelta(days=1)

            date_start = start_date.strftime("%Y-%m-%d")
            date_end = (start_date + timedelta(days=7)).strftime("%Y-%m-%d")

            # For recalls, look for hygienist slots (60 min)
            call_type = global_data.get("call_type", "confirmation")
            if call_type == "recall":
                length_minutes = 60
            else:
                # Try to match original appointment length
                appointment = global_data.get("appointment", {})
                prov_num = appointment.get("ProvNum")
                length_minutes = 30

            slots = od.get_available_slots(
                date_start=date_start,
                date_end=date_end,
                length_minutes=length_minutes,
            )
            # Same-day searches can return openings that already started.
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            slots = [s for s in (slots or []) if s.get("DateTimeStart", "") > now_str]

            if not slots or len(slots) == 0:
                state["outcome"] = "no_slots"
                save_call_state(call_id, state)
                result = SwaigFunctionResult(
                    "I couldn't find any openings in that timeframe. "
                    "We'll call you back with more options, or you can call us anytime."
                )
                _change_step(result, "log_and_end")
                return result

            # Filter by time preference
            filtered = slots
            if time_pref == "morning":
                filtered = [s for s in slots if _get_hour(s.get("DateTimeStart", "")) < 12]
            elif time_pref == "afternoon":
                filtered = [s for s in slots if _get_hour(s.get("DateTimeStart", "")) >= 12]
            if not filtered:
                filtered = slots

            top_slots = filtered[:5]
            summaries = []
            for i, slot in enumerate(top_slots):
                summary = od.format_slot_voice(slot, providers_cache)
                summaries.append(f"Option {i + 1}: {summary}")

            state["available_slots"] = top_slots
            state["slot_summaries"] = summaries
            save_call_state(call_id, state)

            summary_text = " | ".join(summaries)
            result = SwaigFunctionResult(
                f"Found {len(top_slots)} opening{'s' if len(top_slots) > 1 else ''}.\n{summary_text}"
            )
            result.update_global_data({"booking_state": {"slot_summaries": summaries}})
            return result

        # 4. RESCHEDULE OUTBOUND
        @self.tool(
            name="reschedule_outbound",
            description="Reschedule the appointment to a new slot.",
            wait_file="/sounds/typing.mp3",
            function_fillers={"en-US": [
                "Moving your appointment to the new time.",
                "Updating the schedule for you.",
                "Rescheduling that now.",
            ]},
            parameters={
                "type": "object",
                "properties": {
                    "option_number": {
                        "type": "integer",
                        "description": "The slot option number the patient chose (1-5)",
                    },
                },
                "required": ["option_number"],
            },
        )
        def reschedule_outbound(args, raw_data):
            global_data = (raw_data or {}).get("global_data", {})
            appointment = global_data.get("appointment", {})
            apt_num = appointment.get("AptNum")
            option_number = args.get("option_number", 1)
            call_id = _call_id(raw_data)
            state = load_call_state(call_id)

            slots = state.get("available_slots") or []
            idx = option_number - 1
            if idx < 0 or idx >= len(slots):
                return SwaigFunctionResult("Invalid slot selection.")

            new_slot = slots[idx]
            new_datetime = new_slot.get("DateTimeStart", "")
            new_op = new_slot.get("OpNum")
            new_prov = new_slot.get("ProvNum")

            if apt_num:
                od.reschedule_appointment(
                    apt_num=apt_num,
                    new_datetime=new_datetime,
                    new_op=new_op,
                    new_prov_num=new_prov,
                )
                today = datetime.now().strftime("%m/%d")
                od.append_appointment_note(apt_num, f"Rescheduled via AI call {today}")

            provider_name = providers_cache.get(new_prov, "your provider")
            time_display = od.format_datetime_voice(new_datetime)

            state["outcome"] = "rescheduled"
            save_call_state(call_id, state)

            result = SwaigFunctionResult(
                f"Your appointment has been rescheduled to {time_display} with {provider_name}."
            )
            _change_step(result, "log_and_end")
            return result

        # 5. BOOK RECALL
        @self.tool(
            name="book_recall",
            description="Book a recall appointment for the patient.",
            wait_file="/sounds/typing.mp3",
            function_fillers={"en-US": [
                "Booking your cleaning appointment now.",
                "Getting that scheduled for you.",
                "Reserving your spot on the calendar.",
            ]},
            parameters={
                "type": "object",
                "properties": {
                    "option_number": {
                        "type": "integer",
                        "description": "The slot option number the patient chose (1-5)",
                    },
                },
                "required": ["option_number"],
            },
        )
        def book_recall(args, raw_data):
            global_data = (raw_data or {}).get("global_data", {})
            profile = global_data.get("patient_profile", {})
            pat_num = profile.get("PatNum")
            recall_info = global_data.get("recall_info", {})
            option_number = args.get("option_number", 1)
            call_id = _call_id(raw_data)
            state = load_call_state(call_id)

            slots = state.get("available_slots") or []
            idx = option_number - 1
            if idx < 0 or idx >= len(slots):
                return SwaigFunctionResult("Invalid slot selection.")

            selected = slots[idx]
            apt_datetime = selected.get("DateTimeStart", "")
            op_num = selected.get("OpNum")
            prov_num = selected.get("ProvNum")
            recall_type = recall_info.get("recall_type", "Prophy")

            if pat_num:
                od.book_appointment(
                    pat_num=pat_num,
                    op_num=op_num,
                    apt_datetime=apt_datetime,
                    pattern="/XXXXXXXXXX/",  # 60 min for cleaning
                    prov_num=prov_num,
                    note=f"{recall_type} - booked via outbound call",
                )
                od.update_recall_status(
                    pat_num, recall_type=recall_type,
                    status="Scheduled", note="Scheduled via AI outbound call"
                )

            provider_name = providers_cache.get(prov_num, "your provider")
            time_display = od.format_datetime_voice(apt_datetime)

            state["outcome"] = "recall_booked"
            save_call_state(call_id, state)

            result = SwaigFunctionResult(
                f"Your {recall_type.lower()} appointment is booked for {time_display} with {provider_name}."
            )
            _change_step(result, "log_and_end")
            return result

        # 6. DECLINE RECALL
        @self.tool(
            name="decline_recall",
            description="Patient declines to schedule their recall appointment now.",
            wait_file="/sounds/typing.mp3",
            function_fillers={"en-US": [
                "No worries, I'll make a note of that.",
                "Understood, we'll follow up later.",
            ]},
            parameters={"type": "object", "properties": {}, "required": []},
        )
        def decline_recall(args, raw_data):
            global_data = (raw_data or {}).get("global_data", {})
            profile = global_data.get("patient_profile", {})
            pat_num = profile.get("PatNum")
            recall_info = global_data.get("recall_info", {})
            call_id = _call_id(raw_data)
            state = load_call_state(call_id)

            if pat_num:
                od.update_recall_status(
                    pat_num,
                    recall_type=recall_info.get("recall_type", "Prophy"),
                    status="Contacted",
                    note="Patient declined scheduling via AI outbound call",
                )

            state["outcome"] = "recall_declined"
            save_call_state(call_id, state)

            result = SwaigFunctionResult(
                "No problem! We'll remind you again. Feel free to call us anytime to schedule."
            )
            _change_step(result, "log_and_end")
            return result

        # 7. UNABLE TO REACH
        @self.tool(
            name="unable_to_reach_tool",
            description="Log a failed contact attempt (voicemail, no answer, wrong number).",
            wait_file="/sounds/typing.mp3",
            function_fillers={"en-US": [
                "Logging the contact attempt.",
                "Making a note for the office.",
            ]},
            parameters={
                "type": "object",
                "properties": {
                    "reason": {
                        "type": "string",
                        "description": "Reason for failure: voicemail, no_answer, wrong_number",
                        "enum": ["voicemail", "no_answer", "wrong_number"],
                    },
                },
                "required": ["reason"],
            },
        )
        def unable_to_reach_tool(args, raw_data):
            call_id = _call_id(raw_data)
            state = load_call_state(call_id)
            reason = args.get("reason", "no_answer")

            state["outcome"] = f"unable_to_reach:{reason}"
            save_call_state(call_id, state)

            result = SwaigFunctionResult(f"Logged: unable to reach ({reason}).")
            _change_step(result, "log_and_end")
            return result

        # 8. LOG OUTBOUND CALL
        @self.tool(
            name="log_outbound_call",
            description="Log the outbound call result to OpenDental and local DB.",
            wait_file="/sounds/typing.mp3",
            function_fillers={"en-US": [
                "Saving the call details to your chart.",
                "Just wrapping up the paperwork.",
            ]},
            parameters={
                "type": "object",
                "properties": {
                    "outcome": {
                        "type": "string",
                        "description": "Call outcome: confirmed, rescheduled, cancelled, recall_booked, recall_declined, unable_to_reach",
                    },
                    "notes": {
                        "type": "string",
                        "description": "Brief notes about the call",
                    },
                },
                "required": ["outcome"],
            },
        )
        def log_outbound_call(args, raw_data):
            global_data = (raw_data or {}).get("global_data", {})
            profile = global_data.get("patient_profile", {})
            call_id = _call_id(raw_data)
            state = load_call_state(call_id)

            pat_num = profile.get("PatNum")
            outcome = args.get("outcome", state.get("outcome", "unknown"))
            notes = args.get("notes", "")

            summary = f"Outbound AI call - Outcome: {outcome}"
            if notes:
                summary += f" - {notes}"

            if pat_num:
                od.create_commlog(pat_num, summary, mode="Phone")

            save_call_log(
                call_id=call_id,
                pat_num=pat_num,
                patient_name=f"{profile.get('first_name', '')} {profile.get('last_name', '')}".strip(),
                phone="outbound",
                intent=f"outbound_{global_data.get('call_type', 'confirmation')}",
                outcome=outcome,
                apt_num=global_data.get("appointment", {}).get("AptNum"),
                summary=summary,
            )

            return SwaigFunctionResult("Call logged. Goodbye!")


# ── Helpers ──────────────────────────────────────────────────────────


def _get_hour(dt_string):
    """Extract hour from a datetime string."""
    try:
        if "T" in dt_string:
            return int(dt_string.split("T")[1][:2])
        elif " " in dt_string:
            return int(dt_string.split(" ")[1][:2])
    except (ValueError, IndexError):
        pass
    return 12
