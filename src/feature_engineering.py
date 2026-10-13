
""" Labels Trump's posts with GPT: nine answers per post (is it about oil, which way does it push supply, ...).
The answers are saved in data/features.jsonl; pre_process in data_pre_processing.py turns them into model features.

Run from the project root:  python3 src/feature_engineering.py
"""

__author__ = 'Majd Jamal'

import json
import os
from typing import Literal, Optional

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, Field


from data_reader import read_trumps_posts


load_dotenv()   # reads OPENAI_API_KEY from .env in the project folder

MODEL = 'gpt-6-sol'
REASONING = 'low'
MAX_OUTPUT_TOKENS = 1500    # hard cap per post, reasoning included


# The instruction GPT gets with every post.
system_prompt = (
    "Judge one Trump Truth Social post by its words, without hindsight. "
    "Steps: oil-relevant actions, threats, demands, offers, pauses. "
    "Not steps: passed_on text, boasts, vague recaps, goals, others' wishes. "
    "All step fields describe one main step: the first affecting oil supply, shipping or sanctions, else the first; "
    "merely continuing steps rank last. An ultimatum's step is its threat, not its demand."
)

class PostFeatures(BaseModel):
    """ What the model returns for one post.
    """
    text_form: Literal['own_words', 'passed_on'] = Field(description=(
        "passed_on: ignoring URLs and signature, text is only others' words under a label ('From Gingrich:'), "
        "or a link or shared item has only a headline (no I or we), even on actions, or a pointer naming "
        "no action or price ('A MUST READ!'). Otherwise own_words."))

    oil_relevance: Literal['none', 'indirect', 'direct'] = Field(description=(
        "Ignore slogans, endorsements, ended-war boasts. direct: oil or fuel prices, supply, shipping or "
        "facilities (SPR, OPEC, drilling, Hormuz). indirect: a step or current state of a conflict, talks or "
        "deal where a named oil producer (Iran, Russia, Venezuela, Gulf states) is a party, or tariff or trade "
        "steps involving a major economy. none: all else (Gaza, ISIS, drug boats)."))

    price_claim: Literal['up', 'down', 'none'] = Field(description=(
        "Direction Trump says oil, fuel or energy prices are going, will go or must go. A price level or "
        "past comparison ('higher under Biden') alone is none. If he says both up and down, use the present "
        "or future claim, else none. none if passed_on."))

    counterpart: Literal['iran', 'venezuela', 'russia_ukraine', 'gulf_opec', 'china', 'other', 'none'] = Field(description=(
        "The foreign party the main step acts on; if none, the one performing it. If two qualify, take the "
        "value listed first (Venezuela and Iran: iran). Hormuz is iran; OPEC, Iraq and Gulf states are "
        "gulf_opec; Ukraine is russia_ukraine. none: US-only or no step."))

    us_acts: bool = Field(description=(
        "true if Trump or the US takes, orders, threatens, offers or pauses the main step, including joint US "
        "deals or talks, and his threats that name no actor. false for steps by foreign parties or companies, "
        "demands that others act, passive reports, or no step."))

    asset_class: Literal['none_or_unspecified', 'shipping_route', 'oil_supply_asset', 'refinery_or_fuel_retail',
                         'gas_or_power', 'military_leadership_or_other'] = Field(description=(
        "The main step's object; if several, the value listed first. none_or_unspecified: no physical object "
        "(talks, deals, ceasefires, sanctions, tariffs, price demands) or no step. shipping_route: straits, "
        "ports, tankers, blockades. oil_supply_asset: oil fields, terminals, pipelines, output, SPR. "
        "gas_or_power: natural gas, LNG, power plants. military_leadership_or_other: anything else."))

    action_status: Literal['withdrawn_or_suspended', 'ongoing_unchanged', 'ordered_or_implementation_reported',
                           'proposed_or_threatened', 'unclear'] = Field(description=(
        "The main step's status; take the first that fits. withdrawn_or_suspended: cancels, postpones or "
        "pauses a threatened or scheduled step not yet done; not ceasefires or ending something in force. "
        "ongoing_unchanged: something in force stays or continues unchanged ('will remain'). "
        "ordered_or_implementation_reported: ordered, agreed, done or under way. proposed_or_threatened: will, "
        "plans, threats, demands, offers, predictions. unclear: no step."))

    deadline_utc: Optional[str] = Field(description=(
        "Future UTC time (2026-03-23T23:44:53Z) when the main step starts, expires or resumes. Durations "
        "('48 hours') run from published_at_utc. Clock times are Eastern; a day without one ('tonight', "
        "'Tuesday') is 23:59 Eastern. Null for 'today', 'soon', 'until a deal', recalled deadlines, or no step."))

    oil_mechanism: Literal['supply_or_transit_loss', 'supply_or_transit_gain', 'sanctions_tighten', 'sanctions_ease',
                           'producer_price_pressure', 'downstream_or_retail', 'long_run_supply_policy', 'escalation',
                           'deescalation', 'demand_weaker', 'demand_stronger', 'unclear'] = Field(description=(
        "If withdrawn_or_suspended, label the withdrawn step (postponed strikes: escalation; dropped tariffs: demand_weaker). deescalation: "
        "ceasefires, peace talks, peace deals. sanctions_tighten: sanctions or tariffs on Iran, Russia, "
        "Venezuela, OPEC states or their trade partners. demand_weaker: other new tariffs. demand_stronger: "
        "tariff cuts, trade deals, SPR purchases. supply_or_transit_loss: blockades, seized oil unless sold. "
        "long_run_supply_policy: drilling, permits, leases. producer_price_pressure: pushing producers to pump "
        "more. unclear: no step."))

# The same answer format as a JSON schema (web/make_label_spec.py sends it to the website demo).
SCHEMA = PostFeatures.model_json_schema()
SCHEMA['additionalProperties'] = False


def post_message(post):
    """ What GPT reads for one post: its text, publish time and any quoted post.
    """
    quoted = post['quoted_post'] if isinstance(post['quoted_post'], dict) else {}

    return json.dumps({
        'published_at_utc': post['published_at_utc'].strftime('%Y-%m-%dT%H:%M:%SZ'),
        'published_at_et': post['published_at_et'],
        'kind': post['kind'],
        'text': post['text'],
        'shared_content': quoted.get('text'),
    }, ensure_ascii=False)


def make_manifest():


    if os.path.exists('data/manifest.jsonl'):   
        return

    df = read_trumps_posts()
    
    with open('data/manifest.jsonl', 'w') as f:
            for _, post in df.iterrows():
                request = {'custom_id': str(post['id']), 'method': 'POST', 'url': '/v1/responses',
                        'body': {'model': MODEL, 'instructions': system_prompt, 'input': post_message(post),
                                    'reasoning': {'effort': REASONING}, 'max_output_tokens': MAX_OUTPUT_TOKENS,
                                    'text': {'format': {'type': 'json_schema', 'name': 'post_features',
                                                        'schema': SCHEMA, 'strict': True}}}}
                f.write(json.dumps(request, ensure_ascii=False) + '\n')


def start_labeling():

    make_manifest()

    client = OpenAI()

    file = client.files.create(file=open('data/manifest.jsonl', 'rb'), purpose='batch')

    batch = client.batches.create(input_file_id=file.id, endpoint='/v1/responses', completion_window='24h')


    # 4. save its id, to fetch the results later
    with open('data/open_batch.txt', 'w') as f:
        f.write(batch.id)

    print('Batch started:', batch.id)


def collect_features():
    """ Fetches the finished batch and writes one line of labels per post to data/features.jsonl.
    """

    with open('data/open_batch.txt') as f:
        batch_id = f.read().strip()

    client = OpenAI()
    batch = client.batches.retrieve(batch_id)

    if batch.status != 'completed':                 # still running (or failed): try again later
        print('Batch is', batch.status, batch.request_counts)
        return

    results = client.files.content(batch.output_file_id).text.splitlines()

    with open('data/features.jsonl', 'w') as f:

        for line in results:
            result = json.loads(line)
            body = result['response']['body']

            if body['status'] != 'completed':      
                print('Skipped', result['custom_id'], body['status'])
                continue

            text = next(c['text'] for item in body['output'] if item['type'] == 'message'
                        for c in item['content'] if c['type'] == 'output_text')

            row = {'id': result['custom_id'], **json.loads(text),
                   'input_tokens': body['usage']['input_tokens'],
                   'output_tokens': body['usage']['output_tokens'],
                   'reasoning_tokens': body['usage']['output_tokens_details']['reasoning_tokens']}
            
            f.write(json.dumps(row, ensure_ascii=False) + '\n')

    os.remove('data/open_batch.txt')                

if __name__ == '__main__':

    start_labeling()

