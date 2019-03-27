# -*- coding: utf-8 -*-

import pandas as pd
import pathlib


def read_data(filename: str):
    """ Read the json file and convert it into a pd.DataFrame
    preprocessing: format datetime

    Args:
        filename: json filename

    Returns:
        pd.DataFrame
    """
    assert filename.endswith('.json'), 'filename should be a json file'
    assert pathlib.Path(filename).exists(), 'filename does not exists'

    df = pd.read_json(filename)
    df = df.assign(createdAt=pd.to_datetime(df.createdAt),
                   publishedAt=pd.to_datetime(df.publishedAt),
                   updatedAt=pd.to_datetime(df.updatedAt))

    # remove trashed data
    df = df[~df.trashed]

    return df


def extract_responses_by_id(responses: list, key: str='138'):
    """ Extract a specific question

    Args:
        responses: list (example df.iloc[0].responses)
        key: questionId (example '142')

    Returns:
        responses as a string
    """

    response = [x['formattedValue'] for x in responses
                if x['questionId'] == key]
    if len(response):
        return response[0]
    else:
        return None


def get_responses(df: pd.DataFrame):
    """ Extract responses and return a pd.DataFrame
    with columns: authorId, questionId, formattedValue

    Args:
        df: dataframe from read_data
    
    Returns:
        pd.DataFrame with responses
    """
    responses = []
    for i, x in df.iterrows():
        df_tmp = (pd.DataFrame(x.responses).
                  filter(['questionId', 'formattedValue']).
                  assign(authorId=x.authorId))
        responses.append(df_tmp)

    return pd.concat(responses, ignore_index=True)


def get_ids_open_reponses(df: pd.DataFrame):
    """ Return the ids of open questions
    i.e does not have a predefined set of possible responses
    """
    list_questions = df.iloc[0].responses

    ids_open_questions = [x['questionId'] for x in list_questions
                          if x['value'] is None or
                          '{"labels"' not in x['value']]
    return ids_open_questions


def get_open_reponses(df: pd.DataFrame):
    """ Filter the data to only return non empty open responses

    Args:
        df: dataframe from read_data

    Returns:
        pd.DataFrame
    """

    df_open = get_responses(df)
    ids_open = get_ids_open_reponses(df)

    df_open = df_open[df_open.questionId.isin(ids_open)]
    df_open = df_open[~pd.isnull(df_open.formattedValue)]
    return df_open
