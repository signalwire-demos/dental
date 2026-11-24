#!/usr/bin/env python3
"""Dental Voice AI Agent — Appointment booking powered by SignalWire + OpenDental."""

import os
import sys
import json
import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from dotenv import load_dotenv
from signalwire import AgentBase, AgentServer
from signalwire.core.function_result import SwaigFunctionResult

import config
import opendental_client as od
from state_store import (
    load_call_state, save_call_state, delete_call_state,
    cleanup_stale_states, build_ai_summary, save_call_log, get_all_call_logs,
)

load_dotenv()

logging.basicConfig(level=logging.INFO, force=True)
logger = logging.getLogger(__name__)

config.validate()


class DentalAgent(AgentBase):
    """Dental Office Voice AI — Appointments, confirmations, and patient onboarding."""

    def __init__(self):
        super().__init__(name="DentalAgent", route="/swml",
                         record_call=True, record_format="wav", record_stereo=True)

        # AI model
        self.set_param("ai_model", config.AI_MODEL)
        self.set_prompt_llm_params(top_p=config.AI_TOP_P, temperature=config.AI_TEMPERATURE)

        # Personality
        self.prompt_add_section("Personality",
            f"You are a friendly, professional dental office receptionist for {config.PRACTICE_NAME}. "
            "You help callers book, confirm, reschedule, and cancel dental appointments. "
            "Keep it warm and efficient — patients appreciate a calm, helpful voice."
        )

        # Voice behavior
        self.prompt_add_section("Rules", body="", bullets=[
            "This is a PHONE CALL. Keep every response to 1-2 short sentences.",
            "Say times naturally: 'nine thirty AM' not '09:30'. Say dates naturally: 'Tuesday February 26th' not '2026-02-26'.",
            "Use provider titles: 'Dr. Smith' not 'provider 1'. Use first names for hygienists.",
            "Never read raw numbers like PatNum or AptNum to the caller.",
            "If the caller seems unsure, gently offer options or ask clarifying questions.",
        ])

        # Voice
        self.add_language("English", "en-US", "azure.en-US-AvaNeural")
        self.add_hints(["appointment", "cleaning", "checkup", "cavity",
                        "filling", "crown", "extraction", "hygienist",
                        "reschedule", "cancel", "confirm"])

        # Post-prompt
        self.set_post_prompt("Summarize the conversation.")

        # Cache providers on startup (must be before _define_tools)
        self._providers_cache = {}
        self._load_providers_cache()

        # State machine (includes gather steps in correct order)
        self._define_state_machine()

        # SWAIG tools
        self._define_tools()

        # Per-call dynamic config
        self.set_dynamic_config_callback(self._per_call_config)

    def _load_providers_cache(self):
        """Pre-load provider names for voice-friendly slot formatting."""
        try:
            providers = od.get_providers()
            if providers and isinstance(providers, list):
                for p in providers:
                    prov_num = p.get("ProvNum")
                    if prov_num and not p.get("IsHidden"):
                        fname = p.get("FName", "")
                        lname = p.get("LName", "")
                        abbr = p.get("Abbr", "")
                        if lname:
                            self._providers_cache[prov_num] = f"Dr. {lname}"
                        elif abbr:
                            self._providers_cache[prov_num] = abbr
                logger.info(f"Loaded {len(self._providers_cache)} providers")
        except Exception as e:
            logger.warning(f"Could not pre-load providers: {e}")

    # ── State Machine ────────────────────────────────────────────────

    def _define_state_machine(self):
        """Define conversation contexts and steps.

        IMPORTANT: Step definition order matters! gather_info steps with
        completion_action="next_step" transition to the next step in
        definition order, so each gather step MUST be defined immediately
        before its target step.
        """
        contexts = self.define_contexts()
        ctx = contexts.add_context("default")

        # GREETING — bare shell; _per_call_config customizes per caller
        greeting = ctx.add_step("greeting")
        greeting.add_section("Task", "Welcome the caller")
        greeting.set_functions("none")
        greeting.set_valid_steps(["collect_patient", "verify_identity"])

        # ── New Patient Onboarding (gather → save) ──────────────────

        ctx.add_step("collect_patient") \
            .set_text("Collect new patient information.") \
            .set_functions("none") \
            .set_gather_info(
                output_key="patient_answers",
                completion_action="next_step",
                prompt=(
                    f"Say: 'Welcome to {config.PRACTICE_NAME}! "
                    "I'd be happy to help you. Since this is your first time calling, "
                    "let me get a few details.' Then ask each question."
                )
            ) \
            .add_gather_question("first_name", "What is your first name?") \
            .add_gather_question("last_name", "And your last name?") \
            .add_gather_question(
                "date_of_birth",
                "What is your date of birth?",
                confirm=True,
                prompt="Accept natural language but submit in YYYY-MM-DD format"
            ) \
            .add_gather_question(
                "phone_number",
                "What's the best phone number to reach you at?",
                confirm=True,
                prompt="Submit as digits only, 10-digit US number"
            ) \
            .add_gather_question(
                "email",
                "Do you have an email address we can use for appointment reminders?",
                prompt="If they decline or say no, submit 'none'"
            ) \
            .set_valid_steps(["save_patient"])

        # save_patient — MUST be right after collect_patient (next_step target)
        ctx.add_step("save_patient") \
            .add_section("Task", "Save the new patient to OpenDental") \
            .add_bullets("Process", [
                "Call create_patient_tool immediately — patient data is in ${patient_answers}",
            ]) \
            .set_functions(["create_patient_tool"]) \
            .set_valid_steps(["identify_intent"])

        # ── Returning Caller Identity Verification ───────────────────

        verify_identity = ctx.add_step("verify_identity")
        verify_identity.add_section("Task", "Verify the caller's identity by confirming their date of birth")
        verify_identity.add_bullets("Process", [
            "Ask: 'For verification, can you please confirm your date of birth?'",
            "Once they provide it, call verify_identity_tool with the date they gave",
        ])
        verify_identity.set_step_criteria("Caller's identity verified via DOB check")
        verify_identity.set_functions(["verify_identity_tool"])
        verify_identity.set_valid_steps(["identify_intent"])

        # ── Intent Routing ───────────────────────────────────────────

        identify_intent = ctx.add_step("identify_intent")
        identify_intent.add_section("Task", "Determine what the caller needs help with")
        identify_intent.add_bullets("Process", [
            "Ask: 'How can I help you today?'",
            "Listen for: booking a new appointment, confirming an existing one, rescheduling, cancelling, or insurance questions",
            "Once you understand their intent, call route_intent with the appropriate action",
            "If they mention a specific date or reason for visit right away, note it — you'll collect details in the next step",
        ])
        identify_intent.set_step_criteria("Caller's intent identified via route_intent")
        identify_intent.set_functions(["route_intent"])
        identify_intent.set_valid_steps(["collect_appt_prefs", "lookup_appointments", "lookup_insurance"])

        # ── Booking Flow ─────────────────────────────────────────────

        # collect_appt_prefs (gather) — next_step → search_slots
        ctx.add_step("collect_appt_prefs") \
            .set_text("Collect appointment preferences.") \
            .set_functions("none") \
            .set_gather_info(
                output_key="appt_prefs_answers",
                completion_action="next_step",
                prompt="Let's find a good time for your appointment."
            ) \
            .add_gather_question(
                "reason",
                "What's the reason for your visit? For example, a cleaning, checkup, or is something bothering you?",
                prompt="Accept any description. Common types: cleaning, checkup, filling, crown, extraction, toothache, pain."
            ) \
            .add_gather_question(
                "preferred_date",
                "What date works best for you?",
                confirm=True,
                prompt="Accept natural language but submit in YYYY-MM-DD format. Must be a future date."
            ) \
            .add_gather_question(
                "time_preference",
                "Do you prefer morning or afternoon?",
                prompt="Submit exactly: morning, afternoon, or no_preference"
            ) \
            .add_gather_question(
                "provider_preference",
                "Do you have a preferred dentist or hygienist?",
                prompt="Submit the provider name if given, or 'no_preference' if they don't have one."
            ) \
            .set_valid_steps(["search_slots"])

        # search_slots — MUST be right after collect_appt_prefs (next_step target)
        search_slots = ctx.add_step("search_slots")
        search_slots.add_section("Task", "Search for available appointment slots")
        search_slots.add_bullets("Process", [
            "Call search_available_slots to find openings based on the caller's preferences",
        ])
        search_slots.set_step_criteria("Slots found and presented")
        search_slots.set_functions(["search_available_slots"])
        search_slots.set_valid_steps(["present_options", "error_recovery"])

        # PRESENT OPTIONS
        present_options = ctx.add_step("present_options")
        present_options.add_section("Task", "Present available time slots and let the caller pick one")
        present_options.add_bullets("Process", [
            "Read each available slot from booking_state.slot_summaries",
            "Label them 'Option 1' then 'Option 2' then 'Option 3' and so on",
            "Ask which one works for them or if they'd like to try different dates",
            "When the caller picks one, call select_slot with that option_number",
            "If none work, call search_again to try different dates",
        ])
        present_options.set_step_criteria("Caller selects a slot via select_slot or requests new search")
        present_options.set_functions(["select_slot", "search_again"])
        present_options.set_valid_steps(["confirm_booking", "collect_appt_prefs"])

        # CONFIRM BOOKING
        confirm_booking = ctx.add_step("confirm_booking")
        confirm_booking.add_section("Task", "Confirm the appointment details with the caller before booking")
        confirm_booking.add_bullets("Process", [
            "Read back the selected slot: day, time, and provider",
            "Ask: 'Shall I go ahead and book that for you?'",
            "If yes, call book_appointment",
            "If no, go back to present_options",
        ])
        confirm_booking.set_step_criteria("Caller confirms or declines the booking")
        confirm_booking.set_functions(["book_appointment", "decline_booking"])
        confirm_booking.set_valid_steps(["create_appointment", "present_options"])

        # CREATE APPOINTMENT
        create_appointment = ctx.add_step("create_appointment")
        create_appointment.add_section("Task", "Book the appointment and wrap up")
        create_appointment.add_bullets("Process", [
            "The appointment has been booked — read back the confirmation details",
            "Tell the caller their appointment is set",
            "Ask if there's anything else they need",
        ])
        create_appointment.set_functions("none")
        create_appointment.set_valid_steps(["wrap_up", "identify_intent"])

        # ── Confirm / Reschedule / Cancel Flow ───────────────────────

        # LOOKUP APPOINTMENTS
        lookup_appointments = ctx.add_step("lookup_appointments")
        lookup_appointments.add_section("Task", "Look up the caller's upcoming appointments")
        lookup_appointments.add_bullets("Process", [
            "Call get_patient_appointments to fetch their scheduled appointments",
        ])
        lookup_appointments.set_step_criteria("Appointments retrieved")
        lookup_appointments.set_functions(["get_patient_appointments"])
        lookup_appointments.set_valid_steps(["select_appointment", "wrap_up"])

        # SELECT APPOINTMENT
        select_appointment = ctx.add_step("select_appointment")
        select_appointment.add_section("Task", "Have the caller select which appointment to act on")
        select_appointment.add_bullets("Process", [
            "Read the upcoming appointments from booking_state.upcoming_appointments",
            "If there's only one, confirm it directly",
            "If multiple, ask which one they mean",
            "Call select_appointment_tool with the appointment number",
        ])
        select_appointment.set_step_criteria("Appointment selected")
        select_appointment.set_functions(["select_appointment_tool"])
        select_appointment.set_valid_steps(["confirm_appointment", "collect_new_prefs", "confirm_cancel"])

        # CONFIRM APPOINTMENT
        confirm_appointment = ctx.add_step("confirm_appointment")
        confirm_appointment.add_section("Task", "Confirm the appointment for the caller")
        confirm_appointment.add_bullets("Process", [
            "Call confirm_appointment_tool to mark it as confirmed in the system",
        ])
        confirm_appointment.set_step_criteria("Appointment confirmed")
        confirm_appointment.set_functions(["confirm_appointment_tool"])
        confirm_appointment.set_valid_steps(["wrap_up"])

        # ── Reschedule Flow ──────────────────────────────────────────

        # collect_new_prefs (gather) — next_step → search_slots_resched
        ctx.add_step("collect_new_prefs") \
            .set_text("Collect new date preferences for rescheduling.") \
            .set_functions("none") \
            .set_gather_info(
                output_key="resched_prefs_answers",
                completion_action="next_step",
                prompt="Let's find a new time that works better for you."
            ) \
            .add_gather_question(
                "new_preferred_date",
                "What date would you like to reschedule to?",
                confirm=True,
                prompt="Accept natural language but submit in YYYY-MM-DD format. Must be a future date."
            ) \
            .add_gather_question(
                "time_preference",
                "Morning or afternoon?",
                prompt="Submit exactly: morning, afternoon, or no_preference"
            ) \
            .set_valid_steps(["search_slots_resched"])

        # search_slots_resched — MUST be right after collect_new_prefs (next_step target)
        search_slots_resched = ctx.add_step("search_slots_resched")
        search_slots_resched.add_section("Task", "Search for new appointment slots for rescheduling")
        search_slots_resched.add_bullets("Process", [
            "Call search_available_slots to find openings based on the caller's new preferences",
        ])
        search_slots_resched.set_step_criteria("New slots found")
        search_slots_resched.set_functions(["search_available_slots"])
        search_slots_resched.set_valid_steps(["present_resched_options", "error_recovery"])

        # PRESENT RESCHEDULE OPTIONS
        present_resched_options = ctx.add_step("present_resched_options")
        present_resched_options.add_section("Task", "Present new time slots for rescheduling")
        present_resched_options.add_bullets("Process", [
            "Read the available slots",
            "When the caller picks one, call reschedule_appointment_tool",
            "If none work, call search_again to try different dates",
        ])
        present_resched_options.set_step_criteria("Caller picks a new slot")
        present_resched_options.set_functions(["reschedule_appointment_tool", "search_again"])
        present_resched_options.set_valid_steps(["wrap_up", "collect_new_prefs"])

        # ── Cancel Flow ──────────────────────────────────────────────

        confirm_cancel = ctx.add_step("confirm_cancel")
        confirm_cancel.add_section("Task", "Confirm the caller wants to cancel their appointment")
        confirm_cancel.add_bullets("Process", [
            "Read back the appointment details",
            "Ask: 'Are you sure you'd like to cancel this appointment?'",
            "If yes, call cancel_appointment_tool",
            "After cancelling, call get_patient_appointments to check for remaining appointments",
            "If the caller has more appointments, ask if they'd like to cancel another",
            "If no, say 'No problem, your appointment is still on the books' and proceed to wrap_up",
        ])
        confirm_cancel.set_step_criteria("Caller confirms or declines cancellation")
        confirm_cancel.set_functions(["cancel_appointment_tool", "keep_appointment", "get_patient_appointments"])
        confirm_cancel.set_valid_steps(["wrap_up", "lookup_appointments", "select_appointment"])

        # ── Insurance Flow ───────────────────────────────────────────

        lookup_insurance = ctx.add_step("lookup_insurance")
        lookup_insurance.add_section("Task", "Check if the patient has insurance on file")
        lookup_insurance.add_bullets("Process", [
            "Call lookup_insurance_tool to check for existing insurance records",
        ])
        lookup_insurance.set_step_criteria("Insurance lookup complete")
        lookup_insurance.set_functions(["lookup_insurance_tool"])
        lookup_insurance.set_valid_steps(["read_insurance", "collect_insurance"])

        read_insurance = ctx.add_step("read_insurance")
        read_insurance.add_section("Task", "Read back the patient's insurance information")
        read_insurance.add_bullets("Process", [
            "Read the insurance details from booking_state.insurance_info",
            "Tell the caller: their carrier name, plan/group name, member ID, and relationship",
            "Ask: 'Would you like to update your insurance information?'",
            "If yes, proceed to collect_insurance",
            "If no, proceed to wrap_up",
        ])
        read_insurance.set_step_criteria("Insurance info read to caller, update decision made")
        read_insurance.set_functions("none")
        read_insurance.set_valid_steps(["collect_insurance", "wrap_up"])

        # collect_insurance (gather) — next_step → verify_insurance
        ctx.add_step("collect_insurance") \
            .set_text("Collect insurance information from the caller.") \
            .set_functions("none") \
            .set_gather_info(
                output_key="insurance_answers",
                completion_action="next_step",
                prompt=(
                    "Let's get your insurance information on file. "
                    "I'll need a few details from your insurance card."
                )
            ) \
            .add_gather_question(
                "carrier_name",
                "Who is your dental insurance through?",
                prompt="Accept the insurance company name, e.g. Delta Dental, Cigna, Aetna, MetLife, Guardian, etc."
            ) \
            .add_gather_question(
                "subscriber_name",
                "Who is the primary policyholder on the plan?",
                prompt="If the patient says 'me' or 'myself', submit 'self'. Otherwise submit the name of the policyholder."
            ) \
            .add_gather_question(
                "relationship",
                "What is your relationship to the policyholder?",
                prompt="Submit exactly one of: Self, Spouse, Child, Other. If subscriber_name is 'self', auto-submit 'Self'."
            ) \
            .add_gather_question(
                "member_id",
                "What is your member ID? It should be on your insurance card.",
                confirm=True,
                prompt="Accept the member/subscriber ID number. Confirm it back to the caller since it's important."
            ) \
            .add_gather_question(
                "group_number",
                "Do you have a group number on your card?",
                prompt="If they have one, submit it. If they say no or don't know, submit 'none'."
            ) \
            .add_gather_question(
                "subscriber_dob",
                "What is the policyholder's date of birth?",
                confirm=True,
                prompt="Only ask if relationship is NOT 'Self'. If Self, auto-submit the patient's DOB from their profile. Submit in YYYY-MM-DD format."
            ) \
            .add_gather_question(
                "employer",
                "What employer is the insurance plan through?",
                prompt="If they know, submit the employer name. If they don't know or it's individual, submit 'none'."
            ) \
            .set_valid_steps(["verify_insurance"])

        # verify_insurance — MUST be right after collect_insurance (next_step target)
        verify_insurance_step = ctx.add_step("verify_insurance")
        verify_insurance_step.add_section("Task", "Save and verify the collected insurance information")
        verify_insurance_step.add_bullets("Process", [
            "Call save_insurance_tool to execute the full insurance save chain",
            "This will search/create carrier, plan, subscription, and patient plan",
        ])
        verify_insurance_step.set_step_criteria("Insurance saved and verified")
        verify_insurance_step.set_functions(["save_insurance_tool"])
        verify_insurance_step.set_valid_steps(["save_insurance", "error_recovery"])

        save_insurance = ctx.add_step("save_insurance")
        save_insurance.add_section("Task", "Confirm insurance has been saved")
        save_insurance.add_bullets("Process", [
            "Tell the caller their insurance is now on file",
            "Read back: carrier name and member ID for confirmation",
            "If they need to make changes, call update_insurance_tool",
            "Otherwise proceed to wrap_up",
        ])
        save_insurance.set_step_criteria("Insurance confirmed and saved")
        save_insurance.set_functions(["update_insurance_tool"])
        save_insurance.set_valid_steps(["wrap_up"])

        # ── Error Recovery & Wrap Up ─────────────────────────────────

        error_recovery = ctx.add_step("error_recovery")
        error_recovery.add_section("Task", "Handle issues and re-route")
        error_recovery.add_bullets("Process", [
            "If no slots were found, offer to try different dates or a different provider",
            "If an API call failed, apologize and offer to try again or take a message",
            "Call search_again to go back to preference collection",
            "Or call restart to start the whole flow over",
        ])
        error_recovery.set_step_criteria("Recovery action taken")
        error_recovery.set_functions(["search_again", "restart"])
        error_recovery.set_valid_steps(["collect_appt_prefs", "collect_new_prefs", "identify_intent"])

        wrap_up = ctx.add_step("wrap_up")
        wrap_up.add_section("Task", "End the call gracefully")
        wrap_up.add_bullets("Process", [
            "Ask if there's anything else the caller needs",
            "If they need more help, call restart to go back to identify_intent",
            f"Say a brief, warm goodbye: 'Thanks for calling {config.PRACTICE_NAME}! Have a great day.'",
            "Call log_call to record this interaction",
        ])
        wrap_up.set_functions(["log_call", "restart"])
        wrap_up.set_valid_steps(["identify_intent"])

    # ── Per-Call Config (Caller Recognition) ─────────────────────────

    def _per_call_config(self, query_params, body_params, headers, agent):
        """Pre-populate patient data for returning callers."""
        call_data = (body_params or {}).get("call", {})
        caller_phone = call_data.get("caller_id_number", "")

        # Look up patient in OpenDental by phone
        patient = None
        if caller_phone:
            patients = od.lookup_patient_by_phone(caller_phone)
            if patients and len(patients) > 0:
                # Take first match, get full details
                pat_num = patients[0].get("PatNum")
                if pat_num:
                    patient = od.get_patient(pat_num)

        if patient:
            # RETURNING PATIENT — skip onboarding
            pat_num = patient.get("PatNum")
            first_name = patient.get("FName", "")
            last_name = patient.get("LName", "")

            profile = {
                "PatNum": pat_num,
                "first_name": first_name,
                "last_name": last_name,
                "phone": caller_phone,
                "email": patient.get("Email", ""),
                "dob": patient.get("Birthdate", ""),
            }

            agent.update_global_data({
                "patient_profile": profile,
                "is_new_caller": False,
                "caller_phone": caller_phone,
            })

            # Modify greeting for returning patient
            ctx = agent._contexts_builder.get_context("default")
            greeting_step = ctx.get_step("greeting")
            greeting_step.clear_sections()
            greeting_step.set_functions("none")
            greeting_step.set_valid_steps(["verify_identity"])

            greeting_step.add_section("Task", "Welcome returning patient and verify identity")
            greeting_step.add_bullets("Process", [
                f"Say: 'Hi {first_name}, thanks for calling {config.PRACTICE_NAME}!'",
                "Say: 'For verification, can you please confirm your date of birth?'",
                "Proceed to verify_identity",
            ])
            greeting_step.set_step_criteria("Greeting delivered, ready for identity verification")

            # Check for upcoming recalls
            try:
                recalls = od.get_recalls(pat_num)
                if recalls and isinstance(recalls, list):
                    for recall in recalls:
                        due_date = recall.get("DateDue", "")
                        if due_date:
                            try:
                                due = datetime.strptime(due_date, "%Y-%m-%d")
                                if due <= datetime.now() + timedelta(days=30):
                                    agent.update_global_data({
                                        "recall_due": f"Patient is due for a recall/cleaning (due: {due_date})"
                                    })
                                    break
                            except ValueError:
                                pass
            except Exception:
                pass

            # Remove onboarding steps
            for step_name in ["collect_patient", "save_patient"]:
                try:
                    ctx.remove_step(step_name)
                except Exception:
                    pass

            agent.prompt_add_section("Patient Profile", "${global_data.patient_profile}")

        else:
            # NEW CALLER — use onboarding flow
            agent.update_global_data({
                "patient_profile": None,
                "is_new_caller": True,
                "caller_phone": caller_phone,
            })

            ctx = agent._contexts_builder.get_context("default")

            # Modify greeting for new caller
            greeting_step = ctx.get_step("greeting")
            greeting_step.clear_sections()
            greeting_step.set_functions("none")
            greeting_step.set_valid_steps(["collect_patient"])

            greeting_step.add_section("Task", "Welcome new caller and begin onboarding")
            greeting_step.add_bullets("Process", [
                f"Say: 'Thanks for calling {config.PRACTICE_NAME}! I'd be happy to help you.'",
                "Say: 'I don't think we've spoken before — let me get a few details so I can assist you.'",
                "Proceed to collect_patient",
            ])
            greeting_step.set_step_criteria("Greeting delivered, ready for onboarding")

    # ── SWAIG Tools ──────────────────────────────────────────────────

    def _define_tools(self):
        """Define all SWAIG tool functions."""

        def _call_id(raw_data):
            if not isinstance(raw_data, dict):
                return "unknown"
            return raw_data.get("call_id", "unknown")

        def _change_step(result, step):
            logger.info(f"step_change: -> {step}")
            result.swml_change_step(step)

        def _sync_summary(result, state):
            result.update_global_data({"booking_state": build_ai_summary(state)})
            return result

        providers_cache = self._providers_cache

        # 1. ROUTE INTENT
        @self.tool(
            name="route_intent",
            description="Route the caller to the correct flow based on their intent.",
            wait_file="/sounds/typing.mp3",
            fillers={"en-US": [
                "Let me get you to the right place.",
                "Sure, let me pull that up.",
            ]},
            parameters={
                "type": "object",
                "properties": {
                    "intent": {
                        "type": "string",
                        "description": "The caller's intent",
                        "enum": ["book", "confirm", "reschedule", "cancel", "insurance"],
                    },
                },
                "required": ["intent"],
            },
        )
        def route_intent(args, raw_data):
            intent = args.get("intent", "book")
            call_id = _call_id(raw_data)
            state = load_call_state(call_id)
            state["intent"] = intent
            save_call_state(call_id, state)

            if intent == "book":
                result = SwaigFunctionResult("Let's book an appointment.")
                _change_step(result, "collect_appt_prefs")
            elif intent == "insurance":
                result = SwaigFunctionResult("Let me check your insurance information.")
                _sync_summary(result, state)
                _change_step(result, "lookup_insurance")
            else:
                # confirm, reschedule, cancel — all need to look up existing appointments
                result = SwaigFunctionResult(f"Let me look up your appointments.")
                _sync_summary(result, state)
                _change_step(result, "lookup_appointments")

            return result

        # 2. CREATE PATIENT
        @self.tool(
            name="create_patient_tool",
            description="Create a new patient record in the system from the onboarding answers.",
            wait_file="/sounds/typing.mp3",
            fillers={"en-US": [
                "Let me set up your patient profile.",
                "Getting you registered in our system.",
                "Creating your record now.",
            ]},
            parameters={"type": "object", "properties": {}, "required": []},
        )
        def create_patient_tool(args, raw_data):
            global_data = (raw_data or {}).get("global_data", {})
            caller_phone = global_data.get("caller_phone", "")
            answers = global_data.get("patient_answers", {})

            first_name = (answers.get("first_name") or "").strip()
            last_name = (answers.get("last_name") or "").strip()

            if not first_name or not last_name:
                result = SwaigFunctionResult("Missing name. Cannot create patient.")
                _change_step(result, "collect_patient")
                return result

            phone = answers.get("phone_number", caller_phone)
            dob = answers.get("date_of_birth")
            email = answers.get("email")
            if email and email.lower() in ("none", "no", "n/a", "skip"):
                email = None

            # Create patient in OpenDental
            patient = od.create_patient(
                first_name=first_name,
                last_name=last_name,
                phone=phone or caller_phone,
                birthdate=dob,
                email=email,
            )

            if not patient:
                result = SwaigFunctionResult(
                    "I had trouble creating your record. Let me try that again."
                )
                _change_step(result, "collect_patient")
                return result

            pat_num = patient.get("PatNum")
            profile = {
                "PatNum": pat_num,
                "first_name": first_name,
                "last_name": last_name,
                "phone": phone or caller_phone,
                "email": email or "",
                "dob": dob or "",
            }

            call_id = _call_id(raw_data)
            state = load_call_state(call_id)
            state["pat_num"] = pat_num
            save_call_state(call_id, state)

            result = SwaigFunctionResult(
                f"Patient created.\nWelcome {first_name}! How can I help you today?"
            )
            result.update_global_data({
                "patient_profile": profile,
                "is_new_caller": False,
            })
            _change_step(result, "identify_intent")
            return result

        # 3. SEARCH AVAILABLE SLOTS
        @self.tool(
            name="search_available_slots",
            description="Search for available appointment openings based on the caller's preferences.",
            wait_file="/sounds/typing.mp3",
            fillers={"en-US": [
                "Let me check the schedule for you.",
                "Searching for available openings.",
                "Looking at our appointment book.",
            ]},
            parameters={"type": "object", "properties": {}, "required": []},
        )
        def search_available_slots(args, raw_data):
            global_data = (raw_data or {}).get("global_data", {})
            call_id = _call_id(raw_data)
            state = load_call_state(call_id)

            # Read preferences from gather answers
            prefs = global_data.get("appt_prefs_answers") or global_data.get("resched_prefs_answers") or {}
            preferred_date = prefs.get("preferred_date") or prefs.get("new_preferred_date", "")
            time_pref = prefs.get("time_preference", "no_preference")
            provider_pref = prefs.get("provider_preference", "no_preference")
            reason = prefs.get("reason", "")

            state["appointment_prefs"] = {
                "reason": reason,
                "preferred_date": preferred_date,
                "time_preference": time_pref,
                "provider_preference": provider_pref,
            }

            # Determine date range (search 7 days from preferred date)
            try:
                start_date = datetime.strptime(preferred_date, "%Y-%m-%d")
            except (ValueError, TypeError):
                start_date = datetime.now() + timedelta(days=1)

            date_start = start_date.strftime("%Y-%m-%d")
            date_end = (start_date + timedelta(days=7)).strftime("%Y-%m-%d")

            # Determine slot length based on reason
            reason_lower = reason.lower() if reason else ""
            if any(w in reason_lower for w in ["cleaning", "checkup", "recall", "exam"]):
                length_minutes = 60
            elif any(w in reason_lower for w in ["crown", "extraction", "root canal"]):
                length_minutes = 90
            else:
                length_minutes = 30  # default

            # Resolve provider preference
            prov_num = None
            if provider_pref and provider_pref.lower() != "no_preference":
                for pnum, pname in providers_cache.items():
                    if provider_pref.lower() in pname.lower():
                        prov_num = pnum
                        break

            # Call OpenDental
            slots = od.get_available_slots(
                date_start=date_start,
                date_end=date_end,
                prov_num=prov_num,
                length_minutes=length_minutes,
            )
            # Same-day searches can return openings that already started.
            now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            slots = [s for s in (slots or []) if s.get("DateTimeStart", "") > now_str]

            if not slots or (isinstance(slots, list) and len(slots) == 0):
                result = SwaigFunctionResult(
                    f"No openings found between {date_start} and {date_end}."
                    f"{' Try a different date range or provider.' if prov_num else ' Try different dates.'}"
                )
                _change_step(result, "error_recovery")
                return result

            # Filter by time preference
            filtered_slots = slots
            if time_pref == "morning":
                filtered_slots = [s for s in slots
                                  if _get_hour(s.get("DateTimeStart", "")) < 12]
            elif time_pref == "afternoon":
                filtered_slots = [s for s in slots
                                  if _get_hour(s.get("DateTimeStart", "")) >= 12]

            if not filtered_slots:
                filtered_slots = slots  # fall back to all slots if filter empties

            # Take top 5 slots
            top_slots = filtered_slots[:5]

            # Build voice-friendly summaries
            summaries = []
            for i, slot in enumerate(top_slots):
                summary = od.format_slot_voice(slot, providers_cache)
                summaries.append(f"Option {i + 1}: {summary}")

            state["available_slots"] = top_slots
            state["slot_summaries"] = summaries
            save_call_state(call_id, state)

            count = len(top_slots)
            summary_text = " | ".join(summaries)
            result = SwaigFunctionResult(
                f"Found {count} opening{'s' if count > 1 else ''}.\n{summary_text}"
            )
            _sync_summary(result, state)

            # Route to the right present step
            intent = state.get("intent", "book")
            if intent == "reschedule":
                _change_step(result, "present_resched_options")
            else:
                _change_step(result, "present_options")

            return result

        # 4. SELECT SLOT
        @self.tool(
            name="select_slot",
            description="Select one of the available appointment slots.",
            wait_file="/sounds/typing.mp3",
            fillers={"en-US": [
                "Locking in that time slot for you.",
                "Great choice, let me grab that one.",
            ]},
            parameters={
                "type": "object",
                "properties": {
                    "option_number": {
                        "type": "integer",
                        "description": "The option number the caller chose (1-5)",
                    },
                },
                "required": ["option_number"],
            },
        )
        def select_slot(args, raw_data):
            option_number = args.get("option_number", 1)
            call_id = _call_id(raw_data)
            state = load_call_state(call_id)

            slots = state.get("available_slots") or []
            summaries = state.get("slot_summaries") or []

            if not slots:
                result = SwaigFunctionResult("No slots available. Let me search again.")
                _change_step(result, "collect_appt_prefs")
                return result

            idx = option_number - 1
            if idx < 0 or idx >= len(slots):
                available = ", ".join(str(i + 1) for i in range(len(slots)))
                return SwaigFunctionResult(
                    f"Invalid selection. Valid options: {available}."
                )

            selected = slots[idx]
            state["selected_slot"] = {
                "DateTimeStart": selected.get("DateTimeStart"),
                "ProvNum": selected.get("ProvNum"),
                "OpNum": selected.get("OpNum"),
                "summary": summaries[idx] if idx < len(summaries) else "",
            }
            save_call_state(call_id, state)

            result = SwaigFunctionResult(f"Slot selected.\n{summaries[idx] if idx < len(summaries) else 'Selected.'}")
            _sync_summary(result, state)
            _change_step(result, "confirm_booking")
            return result

        # 5. BOOK APPOINTMENT
        @self.tool(
            name="book_appointment",
            description="Book the confirmed appointment slot.",
            wait_file="/sounds/typing.mp3",
            fillers={"en-US": [
                "Booking your appointment now.",
                "Getting that on the schedule for you.",
                "Reserving your spot.",
            ]},
            parameters={"type": "object", "properties": {}, "required": []},
        )
        def book_appointment_tool(args, raw_data):
            global_data = (raw_data or {}).get("global_data", {})
            profile = global_data.get("patient_profile") or {}
            call_id = _call_id(raw_data)
            state = load_call_state(call_id)

            pat_num = profile.get("PatNum") or state.get("pat_num")
            selected = state.get("selected_slot")

            if not pat_num:
                result = SwaigFunctionResult("No patient on file.")
                _change_step(result, "collect_patient")
                return result

            if not selected:
                result = SwaigFunctionResult("No slot selected.")
                _change_step(result, "present_options")
                return result

            apt_datetime = selected.get("DateTimeStart", "")
            op_num = selected.get("OpNum")
            prov_num = selected.get("ProvNum")
            prefs = state.get("appointment_prefs") or {}
            reason = prefs.get("reason", "")

            # Determine if new patient
            is_new = global_data.get("is_new_caller", False)

            # Build pattern based on reason
            reason_lower = reason.lower()
            if any(w in reason_lower for w in ["cleaning", "checkup", "recall", "exam"]):
                pattern = "/XXXXXXXXXX/"   # 60 min
            elif any(w in reason_lower for w in ["crown", "extraction", "root canal"]):
                pattern = "/XXXXXXXXXXXXXXXX/"  # 90 min
            else:
                pattern = "/XXXXXX/"  # 40 min default

            appointment = od.book_appointment(
                pat_num=pat_num,
                op_num=op_num,
                apt_datetime=apt_datetime,
                pattern=pattern,
                prov_num=prov_num,
                is_new_patient=is_new,
                note=f"Booked via phone: {reason}" if reason else "Booked via phone",
            )

            if not appointment:
                result = SwaigFunctionResult(
                    "I wasn't able to book that slot. Let me find another option."
                )
                _change_step(result, "error_recovery")
                return result

            apt_num = appointment.get("AptNum")
            provider_name = providers_cache.get(prov_num, "your provider")
            time_display = od.format_datetime_voice(apt_datetime)

            state["booked_appointment"] = {
                "AptNum": apt_num,
                "datetime": apt_datetime,
                "provider": provider_name,
                "reason": reason,
                "time_display": time_display,
            }
            save_call_state(call_id, state)

            result = SwaigFunctionResult(
                f"Appointment booked!\n"
                f"{time_display} with {provider_name}.\n"
                f"Reason: {reason or 'General visit'}."
            )
            _sync_summary(result, state)
            _change_step(result, "create_appointment")
            return result

        # 6. GET PATIENT APPOINTMENTS
        @self.tool(
            name="get_patient_appointments",
            description="Look up the caller's upcoming scheduled appointments.",
            wait_file="/sounds/typing.mp3",
            fillers={"en-US": [
                "Let me pull up your upcoming appointments.",
                "Checking your schedule now.",
                "Looking up your visits.",
            ]},
            parameters={"type": "object", "properties": {}, "required": []},
        )
        def get_patient_appointments_tool(args, raw_data):
            global_data = (raw_data or {}).get("global_data", {})
            profile = global_data.get("patient_profile") or {}
            call_id = _call_id(raw_data)
            state = load_call_state(call_id)

            pat_num = profile.get("PatNum") or state.get("pat_num")
            if not pat_num:
                result = SwaigFunctionResult("No patient on file.")
                _change_step(result, "wrap_up")
                return result

            appointments = od.get_patient_appointments(pat_num, status="Scheduled")
            if not appointments or (isinstance(appointments, list) and len(appointments) == 0):
                result = SwaigFunctionResult("You don't have any upcoming appointments on file.")
                _change_step(result, "wrap_up")
                return result

            # Build voice-friendly summaries
            apt_summaries = []
            apt_details = []
            for i, apt in enumerate(appointments):
                apt_num = apt.get("AptNum")
                apt_dt = apt.get("AptDateTime", "")
                prov_num = apt.get("ProvNum")
                provider_name = providers_cache.get(prov_num, "your provider")
                note = apt.get("Note", "")
                time_display = od.format_datetime_voice(apt_dt)

                summary = f"Appointment {i + 1}: {time_display} with {provider_name}"
                if note:
                    summary += f" ({note})"
                apt_summaries.append(summary)
                apt_details.append({
                    "AptNum": apt_num,
                    "datetime": apt_dt,
                    "ProvNum": prov_num,
                    "provider": provider_name,
                    "note": note,
                    "summary": summary,
                })

            state["upcoming_appointments"] = apt_details
            save_call_state(call_id, state)

            count = len(apt_summaries)
            intent = state.get("intent", "confirm")

            result_text = f"Found {count} upcoming appointment{'s' if count > 1 else ''}.\n"
            result_text += " | ".join(apt_summaries)

            result = SwaigFunctionResult(result_text)
            result.update_global_data({"upcoming_appointments": apt_summaries})
            _sync_summary(result, state)

            if count == 1:
                # Auto-select the only appointment
                state["selected_apt_num"] = apt_details[0]["AptNum"]
                save_call_state(call_id, state)

                if intent == "confirm":
                    _change_step(result, "confirm_appointment")
                elif intent == "reschedule":
                    _change_step(result, "collect_new_prefs")
                elif intent == "cancel":
                    _change_step(result, "confirm_cancel")
                else:
                    _change_step(result, "select_appointment")
            else:
                _change_step(result, "select_appointment")

            return result

        # 7. SELECT APPOINTMENT (when patient has multiple)
        @self.tool(
            name="select_appointment_tool",
            description="Select which appointment the caller wants to act on.",
            wait_file="/sounds/typing.mp3",
            fillers={"en-US": [
                "Pulling up that appointment.",
                "Let me get the details on that one.",
            ]},
            parameters={
                "type": "object",
                "properties": {
                    "appointment_number": {
                        "type": "integer",
                        "description": "Which appointment number the caller chose (1, 2, 3, etc.)",
                    },
                },
                "required": ["appointment_number"],
            },
        )
        def select_appointment_tool(args, raw_data):
            apt_number = args.get("appointment_number", 1)
            call_id = _call_id(raw_data)
            state = load_call_state(call_id)

            upcoming = state.get("upcoming_appointments") or []
            idx = apt_number - 1

            if idx < 0 or idx >= len(upcoming):
                available = ", ".join(str(i + 1) for i in range(len(upcoming)))
                return SwaigFunctionResult(f"Invalid selection. Valid: {available}.")

            selected_apt = upcoming[idx]
            state["selected_apt_num"] = selected_apt["AptNum"]
            save_call_state(call_id, state)

            intent = state.get("intent", "confirm")
            result = SwaigFunctionResult(f"Selected: {selected_apt['summary']}")
            _sync_summary(result, state)

            if intent == "confirm":
                _change_step(result, "confirm_appointment")
            elif intent == "reschedule":
                _change_step(result, "collect_new_prefs")
            elif intent == "cancel":
                _change_step(result, "confirm_cancel")

            return result

        # 8. CONFIRM APPOINTMENT
        @self.tool(
            name="confirm_appointment_tool",
            description="Mark the selected appointment as confirmed.",
            wait_file="/sounds/typing.mp3",
            fillers={"en-US": [
                "Marking your appointment as confirmed.",
                "Updating your confirmation status.",
                "Locking that in for you.",
            ]},
            parameters={"type": "object", "properties": {}, "required": []},
        )
        def confirm_appointment_tool(args, raw_data):
            call_id = _call_id(raw_data)
            state = load_call_state(call_id)
            apt_num = state.get("selected_apt_num")

            if not apt_num:
                return SwaigFunctionResult("No appointment selected.")

            confirmed = od.confirm_appointment(apt_num, confirm_val="Confirmed")

            if confirmed:
                result = SwaigFunctionResult("Your appointment has been confirmed! We'll see you then.")
            else:
                result = SwaigFunctionResult("I had trouble confirming that. But don't worry, your appointment is still scheduled.")

            _sync_summary(result, state)
            _change_step(result, "wrap_up")
            return result

        # 9. RESCHEDULE APPOINTMENT
        @self.tool(
            name="reschedule_appointment_tool",
            description="Reschedule the appointment to a new time slot.",
            wait_file="/sounds/typing.mp3",
            fillers={"en-US": [
                "Moving your appointment to the new time.",
                "Rescheduling that for you now.",
                "Updating the schedule.",
            ]},
            parameters={
                "type": "object",
                "properties": {
                    "option_number": {
                        "type": "integer",
                        "description": "The new slot option number the caller chose",
                    },
                },
                "required": ["option_number"],
            },
        )
        def reschedule_appointment_tool(args, raw_data):
            option_number = args.get("option_number", 1)
            call_id = _call_id(raw_data)
            state = load_call_state(call_id)

            apt_num = state.get("selected_apt_num")
            slots = state.get("available_slots") or []
            summaries = state.get("slot_summaries") or []

            if not apt_num:
                return SwaigFunctionResult("No appointment selected to reschedule.")

            idx = option_number - 1
            if idx < 0 or idx >= len(slots):
                return SwaigFunctionResult("Invalid slot selection.")

            new_slot = slots[idx]
            new_datetime = new_slot.get("DateTimeStart", "")
            new_op = new_slot.get("OpNum")
            new_prov = new_slot.get("ProvNum")

            rescheduled = od.reschedule_appointment(
                apt_num=apt_num,
                new_datetime=new_datetime,
                new_op=new_op,
                new_prov_num=new_prov,
            )

            if rescheduled:
                provider_name = providers_cache.get(new_prov, "your provider")
                time_display = od.format_datetime_voice(new_datetime)
                result = SwaigFunctionResult(
                    f"Appointment rescheduled!\nNew time: {time_display} with {provider_name}."
                )
            else:
                result = SwaigFunctionResult("I had trouble rescheduling. Let me try finding another slot.")
                _change_step(result, "error_recovery")
                return result

            _sync_summary(result, state)
            _change_step(result, "wrap_up")
            return result

        # 10. CANCEL APPOINTMENT
        @self.tool(
            name="cancel_appointment_tool",
            description="Cancel the selected appointment.",
            wait_file="/sounds/typing.mp3",
            fillers={"en-US": [
                "Processing your cancellation.",
                "Removing that from the schedule.",
            ]},
            parameters={"type": "object", "properties": {}, "required": []},
        )
        def cancel_appointment_tool(args, raw_data):
            call_id = _call_id(raw_data)
            state = load_call_state(call_id)
            apt_num = state.get("selected_apt_num")

            if not apt_num:
                return SwaigFunctionResult("No appointment selected to cancel.")

            cancelled = od.cancel_appointment(apt_num)

            if cancelled:
                result = SwaigFunctionResult("Your appointment has been cancelled. If you need to rebook, just give us a call anytime.")
            else:
                result = SwaigFunctionResult("I had trouble cancelling that appointment. Please call back and we'll sort it out.")

            _sync_summary(result, state)
            _change_step(result, "wrap_up")
            return result

        # 11. KEEP APPOINTMENT (decline cancel)
        @self.tool(
            name="keep_appointment",
            description="Caller decided not to cancel after all.",
            wait_file="/sounds/typing.mp3",
            fillers={"en-US": [
                "No worries, keeping everything as is.",
                "Good call, we'll keep you on the books.",
            ]},
            parameters={"type": "object", "properties": {}, "required": []},
        )
        def keep_appointment(args, raw_data):
            result = SwaigFunctionResult("No problem! Your appointment is still on the books.")
            _change_step(result, "wrap_up")
            return result

        # 12. DECLINE BOOKING (go back to options)
        @self.tool(
            name="decline_booking",
            description="Caller doesn't want to book this slot — go back to options.",
            wait_file="/sounds/typing.mp3",
            fillers={"en-US": [
                "No problem, let me pull up the other options.",
                "Sure, let's find a better fit.",
            ]},
            parameters={"type": "object", "properties": {}, "required": []},
        )
        def decline_booking(args, raw_data):
            result = SwaigFunctionResult("No problem, let's look at those options again.")
            _change_step(result, "present_options")
            return result

        # 13. SEARCH AGAIN
        @self.tool(
            name="search_again",
            description="Caller wants to try different dates or preferences.",
            wait_file="/sounds/typing.mp3",
            fillers={"en-US": [
                "Let me search for some different options.",
                "Sure, let's try other dates.",
            ]},
            parameters={"type": "object", "properties": {}, "required": []},
        )
        def search_again(args, raw_data):
            call_id = _call_id(raw_data)
            state = load_call_state(call_id)
            intent = state.get("intent", "book")

            result = SwaigFunctionResult("Let's try different dates.")
            if intent == "reschedule":
                _change_step(result, "collect_new_prefs")
            else:
                _change_step(result, "collect_appt_prefs")
            return result

        # 14. RESTART
        @self.tool(
            name="restart",
            description="Start the conversation flow over — caller needs something else.",
            wait_file="/sounds/typing.mp3",
            fillers={"en-US": [
                "Of course, let me help with that.",
                "Sure thing, what else do you need?",
            ]},
            parameters={"type": "object", "properties": {}, "required": []},
        )
        def restart(args, raw_data):
            result = SwaigFunctionResult("Of course! What else can I help you with?")
            _change_step(result, "identify_intent")
            return result

        # 15. LOG CALL
        @self.tool(
            name="log_call",
            description="Log this call to the patient's chart. Call this before ending.",
            wait_file="/sounds/typing.mp3",
            fillers={"en-US": [
                "Just saving some notes to your chart.",
                "Logging our conversation.",
            ]},
            parameters={
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "Brief summary of what happened on this call",
                    },
                },
                "required": ["summary"],
            },
        )
        def log_call(args, raw_data):
            global_data = (raw_data or {}).get("global_data", {})
            profile = global_data.get("patient_profile") or {}
            call_id = _call_id(raw_data)
            state = load_call_state(call_id)
            summary = args.get("summary", "Phone call")

            pat_num = profile.get("PatNum") or state.get("pat_num")

            # Log to OpenDental
            if pat_num:
                od.create_commlog(pat_num, summary, mode="Phone")

            # Log locally
            save_call_log(
                call_id=call_id,
                pat_num=pat_num,
                patient_name=f"{profile.get('first_name', '')} {profile.get('last_name', '')}".strip(),
                phone=global_data.get("caller_phone", ""),
                intent=state.get("intent"),
                outcome=summary,
                apt_num=state.get("booked_appointment", {}).get("AptNum") if state.get("booked_appointment") else state.get("selected_apt_num"),
                summary=summary,
            )

            return SwaigFunctionResult("Call logged.")

        # 16. VERIFY IDENTITY (DOB check for returning callers)
        @self.tool(
            name="verify_identity_tool",
            description="Verify the caller's identity by checking their date of birth against the record.",
            wait_file="/sounds/typing.mp3",
            fillers={"en-US": [
                "Let me verify that against your records.",
                "Checking your date of birth now.",
                "Just confirming your identity.",
            ]},
            parameters={
                "type": "object",
                "properties": {
                    "date_of_birth": {
                        "type": "string",
                        "description": "The date of birth provided by the caller, in YYYY-MM-DD format",
                    },
                },
                "required": ["date_of_birth"],
            },
        )
        def verify_identity_tool(args, raw_data):
            global_data = (raw_data or {}).get("global_data", {})
            profile = global_data.get("patient_profile") or {}
            provided_dob = args.get("date_of_birth", "")

            stored_dob = profile.get("dob", "")

            if not stored_dob:
                result = SwaigFunctionResult("Verified. How can I help you today?")
                _change_step(result, "identify_intent")
                return result

            # Normalize dates for comparison
            try:
                provided = datetime.strptime(provided_dob, "%Y-%m-%d").date()
                stored = datetime.strptime(stored_dob, "%Y-%m-%d").date()
                match = provided == stored
            except (ValueError, TypeError):
                # Fuzzy match — compare strings
                match = provided_dob.strip() == stored_dob.strip()

            if match:
                first_name = profile.get("first_name", "")
                result = SwaigFunctionResult(
                    f"Identity verified. Thank you, {first_name}. How can I help you today?"
                )
                _change_step(result, "identify_intent")
            else:
                result = SwaigFunctionResult(
                    "That doesn't match what we have on file. "
                    "Could you try again? What is your date of birth?"
                )
                # Stay on verify_identity step — don't change step

            return result

        # 17. LOOKUP INSURANCE (renumbered from 16)
        @self.tool(
            name="lookup_insurance_tool",
            description="Check if the patient has insurance on file.",
            wait_file="/sounds/typing.mp3",
            fillers={"en-US": [
                "Let me check your insurance on file.",
                "Looking up your dental coverage.",
                "Pulling up your insurance details.",
            ]},
            parameters={"type": "object", "properties": {}, "required": []},
        )
        def lookup_insurance_tool(args, raw_data):
            global_data = (raw_data or {}).get("global_data", {})
            profile = global_data.get("patient_profile") or {}
            call_id = _call_id(raw_data)
            state = load_call_state(call_id)

            pat_num = profile.get("PatNum") or state.get("pat_num")
            if not pat_num:
                result = SwaigFunctionResult("No patient on file. Let me get your information first.")
                _change_step(result, "collect_patient")
                return result

            insurance = od.get_patient_insurance(pat_num)

            if insurance and isinstance(insurance, list) and len(insurance) > 0:
                # Patient has insurance on file
                primary = insurance[0]
                carrier_name = primary.get("CarrierName", "Unknown")
                group_name = primary.get("GroupName", "")
                member_id = primary.get("SubscriberID", "")
                relationship = primary.get("Relationship", "Self")
                plan_type = primary.get("PlanType", "")

                state["insurance_info"] = {
                    "found": True,
                    "carrier_name": carrier_name,
                    "group_name": group_name,
                    "member_id": member_id,
                    "relationship": relationship,
                    "plan_type": plan_type,
                    "pat_plan_num": primary.get("PatPlanNum"),
                    "plan_num": primary.get("PlanNum"),
                    "ins_sub_num": primary.get("InsSubNum"),
                    "carrier_num": primary.get("CarrierNum"),
                    "all_plans": insurance,
                }
                save_call_state(call_id, state)

                result = SwaigFunctionResult(
                    f"Insurance found on file.\n"
                    f"Carrier: {carrier_name}\n"
                    f"Plan: {group_name}\n"
                    f"Member ID: {member_id}\n"
                    f"Relationship: {relationship}"
                )
                _sync_summary(result, state)
                _change_step(result, "read_insurance")
            else:
                # No insurance on file
                state["insurance_info"] = {"found": False}
                save_call_state(call_id, state)

                result = SwaigFunctionResult(
                    "No insurance found on file. Let me collect your insurance information."
                )
                _sync_summary(result, state)
                _change_step(result, "collect_insurance")

            return result

        # 18. SAVE INSURANCE
        @self.tool(
            name="save_insurance_tool",
            description="Save insurance information to the system. Executes the full chain: find/create carrier, plan, subscription, and patient plan.",
            wait_file="/sounds/typing.mp3",
            fillers={"en-US": [
                "Saving your insurance information now.",
                "Setting up your coverage in our system.",
                "Processing your insurance details, this may take a moment.",
            ]},
            parameters={"type": "object", "properties": {}, "required": []},
        )
        def save_insurance_tool(args, raw_data):
            global_data = (raw_data or {}).get("global_data", {})
            profile = global_data.get("patient_profile") or {}
            answers = global_data.get("insurance_answers", {})
            call_id = _call_id(raw_data)
            state = load_call_state(call_id)

            pat_num = profile.get("PatNum") or state.get("pat_num")
            if not pat_num:
                result = SwaigFunctionResult("No patient on file.")
                _change_step(result, "collect_patient")
                return result

            carrier_name = (answers.get("carrier_name") or "").strip()
            member_id = (answers.get("member_id") or "").strip()
            group_number = answers.get("group_number", "")
            if group_number and group_number.lower() in ("none", "no", "n/a"):
                group_number = ""
            relationship = answers.get("relationship", "Self")
            employer = answers.get("employer", "")
            if employer and employer.lower() in ("none", "no", "n/a", "unknown"):
                employer = ""

            if not carrier_name:
                result = SwaigFunctionResult("Missing carrier name. Let me collect that again.")
                _change_step(result, "collect_insurance")
                return result

            if not member_id:
                result = SwaigFunctionResult("Missing member ID. Let me collect that again.")
                _change_step(result, "collect_insurance")
                return result

            # Step 1: Find or create carrier
            carriers = od.search_carriers(carrier_name)
            if carriers and len(carriers) > 0:
                carrier_num = carriers[0].get("CarrierNum")
                logger.info(f"Found existing carrier: {carrier_name} (CarrierNum={carrier_num})")
            else:
                carrier = od.create_carrier(carrier_name)
                if not carrier:
                    result = SwaigFunctionResult("I had trouble saving your insurance carrier. Let me try again.")
                    _change_step(result, "error_recovery")
                    return result
                carrier_num = carrier.get("CarrierNum")
                logger.info(f"Created new carrier: {carrier_name} (CarrierNum={carrier_num})")

            # Step 2: Find or create insurance plan
            plans = od.search_insurance_plans(carrier_num)
            plan_num = None
            if plans:
                # Try to match by group number
                for p in plans:
                    if group_number and p.get("GroupNum") == group_number:
                        plan_num = p.get("PlanNum")
                        break
                if not plan_num:
                    plan_num = plans[0].get("PlanNum")

            if not plan_num:
                plan = od.create_insurance_plan(
                    carrier_num=carrier_num,
                    group_name=employer or carrier_name,
                    group_num=group_number or None,
                )
                if not plan:
                    result = SwaigFunctionResult("I had trouble saving your insurance plan. Let me try again.")
                    _change_step(result, "error_recovery")
                    return result
                plan_num = plan.get("PlanNum")

            # Step 3: Create insurance subscription
            subscriber_pat_num = pat_num  # Default: patient is subscriber
            sub = od.create_insurance_sub(
                plan_num=plan_num,
                subscriber_pat_num=subscriber_pat_num,
                subscriber_id=member_id,
            )
            if not sub:
                result = SwaigFunctionResult("I had trouble linking your subscription. Let me try again.")
                _change_step(result, "error_recovery")
                return result
            ins_sub_num = sub.get("InsSubNum")

            # Step 4: Create patient plan
            pat_plan = od.create_patient_plan(
                pat_num=pat_num,
                ins_sub_num=ins_sub_num,
                ordinal=1,
                relationship=relationship,
            )
            if not pat_plan:
                result = SwaigFunctionResult("I had trouble assigning your insurance. Let me try again.")
                _change_step(result, "error_recovery")
                return result
            pat_plan_num = pat_plan.get("PatPlanNum")

            # Step 5: Verify insurance
            od.verify_insurance(pat_plan_num)

            state["insurance_info"] = {
                "found": True,
                "carrier_name": carrier_name,
                "member_id": member_id,
                "group_number": group_number,
                "relationship": relationship,
                "pat_plan_num": pat_plan_num,
                "plan_num": plan_num,
                "ins_sub_num": ins_sub_num,
                "carrier_num": carrier_num,
                "saved": True,
            }
            save_call_state(call_id, state)

            result = SwaigFunctionResult(
                f"Insurance saved!\n"
                f"Carrier: {carrier_name}\n"
                f"Member ID: {member_id}\n"
                f"{'Group: ' + group_number if group_number else ''}"
            )
            _sync_summary(result, state)
            _change_step(result, "save_insurance")
            return result

        # 19. UPDATE INSURANCE
        @self.tool(
            name="update_insurance_tool",
            description="Update existing insurance information — for plan changes, new card, etc.",
            wait_file="/sounds/typing.mp3",
            fillers={"en-US": [
                "Let me update your insurance information.",
                "Pulling up your plan to make changes.",
            ]},
            parameters={"type": "object", "properties": {}, "required": []},
        )
        def update_insurance_tool(args, raw_data):
            call_id = _call_id(raw_data)
            state = load_call_state(call_id)

            result = SwaigFunctionResult("Let me collect your updated insurance information.")
            _sync_summary(result, state)
            _change_step(result, "collect_insurance")
            return result

    # ── Lifecycle ────────────────────────────────────────────────────

    def _render_swml(self, call_id=None, modifications=None):
        """Override to dump generated SWML to stderr for debugging."""
        swml = super()._render_swml(call_id, modifications)
        try:
            parsed = json.loads(swml) if isinstance(swml, str) else swml
            print(json.dumps(parsed, indent=2, default=str), file=sys.stderr)
        except Exception:
            print(swml, file=sys.stderr)
        return swml

    def on_summary(self, summary=None, raw_data=None):
        """Called when the post-prompt summary is received after the call ends."""
        if summary:
            logger.info(f"Call summary: {summary}")

        if raw_data:
            calls_dir = Path(__file__).parent / "calls"
            calls_dir.mkdir(exist_ok=True)
            call_id = raw_data.get("call_id", "unknown")
            out_path = calls_dir / f"{call_id}.json"
            try:
                out_path.write_text(json.dumps(raw_data, indent=2, default=str))
                logger.info(f"Saved call data to {out_path}")
            except Exception as e:
                logger.error(f"Failed to save call data: {e}")

            delete_call_state(call_id)
            cleanup_stale_states(24)


# ── Helpers ──────────────────────────────────────────────────────────


def _get_hour(dt_string):
    """Extract hour from a datetime string for morning/afternoon filtering."""
    try:
        if "T" in dt_string:
            return int(dt_string.split("T")[1][:2])
        elif " " in dt_string:
            return int(dt_string.split(" ")[1][:2])
    except (ValueError, IndexError):
        pass
    return 12  # default to noon if parsing fails


# ── Server Setup ─────────────────────────────────────────────────────


def print_startup_url():
    """Print the full SWML URL with auth for easy copy/paste."""
    base = config.SWML_PROXY_URL_BASE
    if base:
        base = base.rstrip("/")
    else:
        host = config.HOST if config.HOST != "0.0.0.0" else "localhost"
        base = f"http://{host}:{config.PORT}"

    user = config.SWML_BASIC_AUTH_USER
    password = config.SWML_BASIC_AUTH_PASSWORD

    if user and password:
        scheme, rest = base.split("://", 1)
        url = f"{scheme}://{user}:{password}@{rest}/swml"
    else:
        url = f"{base}/swml"

    logger.info(f"SWML endpoint: {url}")


def create_server():
    """Create and configure the AgentServer."""
    server = AgentServer(host=config.HOST, port=config.PORT)
    server.register(DentalAgent(), "/swml")

    # Register outbound agent
    from outbound_agent import OutboundConfirmAgent
    server.register(OutboundConfirmAgent(), "/outbound")

    # Import confirmation runner
    from confirmation_runner import runner as confirmation_runner, start_scheduler

    @server.app.get("/api/phone")
    def get_phone():
        return {
            "phone": config.SIGNALWIRE_PHONE_NUMBER,
            "display": config.DISPLAY_PHONE_NUMBER or config.SIGNALWIRE_PHONE_NUMBER,
        }

    @server.app.get("/api/calls")
    def api_calls():
        return {"calls": get_all_call_logs()}

    @server.app.get("/api/confirmations")
    def api_confirmations():
        """List today's confirmation queue with status."""
        return confirmation_runner.get_status()

    @server.app.post("/api/confirmations/run")
    def api_confirmations_run():
        """Manually trigger a confirmation run."""
        if confirmation_runner.is_running:
            return {"status": "already_running"}
        import threading
        thread = threading.Thread(target=confirmation_runner.run, daemon=True)
        thread.start()
        return {"status": "started"}

    @server.app.get("/api/recalls")
    def api_recalls():
        """List patients due for recall."""
        from datetime import timedelta as td
        today = datetime.now().date()
        start = today.strftime("%Y-%m-%d")
        end = (today + td(days=config.RECALL_WINDOW_DAYS)).strftime("%Y-%m-%d")
        recalls = od.get_recall_list(start, end)
        return {"recalls": recalls or []}

    # Serve static files (sounds, etc.) from web/ directory
    web_dir = Path(__file__).parent / "web"
    if web_dir.exists():
        server.serve_static_files(str(web_dir))

    # Start outbound scheduler if enabled
    start_scheduler()

    print_startup_url()
    return server


server = create_server()
app = server.app

if __name__ == "__main__":
    server.run()
