# defining API key, instantiating MediaCloud API
def init_mc_api(api_key=None):
    
    # importing necessary modules
    from dotenv import load_dotenv
    import os
    import mediacloud.api
    
    # loading environment variables from .env file
    load_dotenv()
    
    # if no API key is passed, function assumes API key is defined in .env file as MC_API_KEY
    if not api_key:
        api_key = os.getenv("MC_API_KEY")
    
    # returning API instance
    return mediacloud.api.MediaCloud(api_key)

# formatting date ranges for MediaCloud API calls.
# required arguments: MediaCloud API instance (from init_mc_api) and date range, eg ["8/1/2021", "9/30/2021"]
def clean_api_date(mediacloud_api, date_range, verbose=False):
    
    # importing necessary modules
    from datetime import datetime
    from dateutil import parser
    
    # making sure two dates are passed into the function
    assert len(date_range) == 2, "Please provide both a start and end date for the date interval."
    
    # parsing dates, returning API date range clause
    start_date = parser.parse(date_range[0])
    end_date = parser.parse(date_range[1])
    if verbose:
        print(f"Date range: between {start_date.strftime('%m/%d/%Y')} and {end_date.strftime('%m/%d/%Y')}")
    return mediacloud_api.dates_as_query_clause(start_date, end_date)

# ensuring that the sources passed into functions are of the correct format
# dict, with format {source_name: MediaCloud_ID}
# setting relevant query parameter for individual source vs MediaCloud collection search
def check_source_type(sources, source_type):
    assert type(sources) == dict,\
    "Please provide the sources in a dict, in the format <Source Name>: <MediaCloud ID>"
    
    assert source_type in {"media", "collection"},\
    'Please specify either "media" or "collection" as source_type.'
    
    if source_type == "media":
        mc_source_type = "media_id"
    elif source_type == "collection":
        mc_source_type = "tags_id_media"
    return mc_source_type

# building a function to search a string among the sources given
def search_sources(query, sources, source_type="media", date_range=None,
                   query_context=None, api_key=None, verbose=False, urls=False):
    
    # initializing MediaCloud API, checking source format, cleaning date_range
    mc = init_mc_api(api_key=api_key)
    mc_source_type = check_source_type(sources, source_type)
    api_date_range = clean_api_date(mc, date_range) if date_range else None
    
    # ensuring the query is a string
    query = str(query)
    print(f"Query: {query}")
    
    # initializing dataframe to store the query data
    import pandas as pd
    import numpy as np
    if urls:
        story_counts = pd.DataFrame(columns=["Name", "Relevant Stories", "Total Stories", "Attention (%)", "Story URLs"])
    else:
        story_counts = pd.DataFrame(columns=["Name", "Relevant Stories", "Total Stories", "Attention (%)"])
    story_counts.index.name = "MediaCloud ID"
    
    # going through each source and querying relevant and total stories
    for source_name in sources:
        if verbose:
            print(f"{source_name}:")
        
        # defining overall context and specific query for stories
        if query_context:
            total_query = f'({query_context}) and {mc_source_type}:{sources[source_name]}'
        else:
            total_query = f'{mc_source_type}:{sources[source_name]}'
        api_query = f'({query}) and {total_query}'

        # API calls to count relevant and total stories
        relevant_stories = mc.storyCount(api_query, api_date_range)['count']
        total_stories = mc.storyCount(total_query, api_date_range)['count']
        
        # getting urls for a sample of 20 stories relevant to the query
        if urls:
            stories = mc.storyList(api_query, api_date_range, sort=mc.SORT_RANDOM)
            story_urls = [story['url'] for story in stories]
        
        # appending data to dataframe
        try:
            attention = (relevant_stories / total_stories) * 100
        except ZeroDivisionError:
            attention = np.nan
        if urls:
            story_counts.loc[sources[source_name]] = [source_name, relevant_stories, total_stories, attention, story_urls]
        else:
            story_counts.loc[sources[source_name]] = [source_name, relevant_stories, total_stories, attention]
        
        # printing story count and attention
        if verbose:
            print(f"{relevant_stories} stories about {query}, {total_stories} total")
            if not np.isnan(attention):
                print(f"{attention}% of stories are about {query}\n")
            else:
                print("")
    
    return story_counts

# calculating number of stories that mention some topics and keywords we're interested in
def calculate_percentages(keywords, sources, source_type="media", keyword_labels=None,
                          date_range=None, query_context=None, api_key=None):
    
    # initializing MediaCloud API, checking source format, cleaning date_range
    mc = init_mc_api(api_key=api_key)
    mc_source_type = check_source_type(sources, source_type)
    api_date_range = clean_api_date(mc, date_range) if date_range else None
    
    # formatting media_ids for API query
    media_ids = list(sources.values())
    api_media_ids = " OR ".join(f"{mc_source_type}:{media_id}" for media_id in media_ids)

    # defining overall context and specific query for stories
    if query_context:
        total_query = f'({query_context}) and ({api_media_ids})'
    else:
        total_query = f'({api_media_ids})'
    
    # joining keywords to calculate aggregate percentages
    all_keywords = " OR ".join(keywords)

    # adding queries to list to loop through later
    keyword_queries = [f'({keyword}) and {total_query}' for keyword in keywords+[f"({all_keywords})"]]
    
    # story count for query context
    total_results = mc.storyCount(total_query, api_date_range)['count']

    # keyword story counts
    keyword_results = [mc.storyCount(keyword_query, api_date_range)['count'] for keyword_query in keyword_queries]
    
    # calculating percentages of stories that mention keywords
    import numpy as np
    keyword_percentages = np.divide(keyword_results, total_results) * 100
    keyword_percentages = np.around(keyword_percentages, decimals=2)
    
    # printing percentages
    print("Percentage of stories within specified context that mention:\n")
    labels = keyword_labels if keyword_labels else keywords
    for i in range(len(keywords)):
        print(f"{labels[i]}: {keyword_percentages[i]}%\n")
    print(f"All Keywords: {keyword_percentages[-1]}%")
    
    return keyword_percentages

# adapting simple_word_cloud() function from Laura's previous code
def word_cloud(query, sources, date_range, source_type="media", 
               save_img=False, filename=None, custom_stopwords=None, api_key=None, verbose=False):
    
    # initializing MediaCloud API, checking source format, cleaning date_range
    mc = init_mc_api(api_key=api_key)
    mc_source_type = check_source_type(sources, source_type)
    api_date_range = clean_api_date(mc, date_range) if date_range else None
    
    # formatting query, sources, and date_range for API
    assert type(query) == str, "Please input a string as the query."
    media_ids = list(sources.values())
    api_media_ids = " OR ".join([f"{mc_source_type}:{media_id}" for media_id in media_ids])
    
    api_query = f"({query}) and ({api_media_ids})"
    if verbose:
        print(f"Query: {api_query}")

    # building term/document matrix, separating word list from word matrix
    story_count = mc.storyCount(api_query, api_date_range)["count"]
    doc_term_matrix = mc.storyWordMatrix(api_query, api_date_range, rows=story_count, max_words=100)
    word_list = doc_term_matrix["word_list"]
    word_matrix = doc_term_matrix["word_matrix"]
    top_words = [word[1] for word in word_list]
    
    # aggregating word frequencies in each document
    word_freqs = {}
    for word_id in range(len(top_words)):
        word_freqs[word_id] = 0
        for story in word_matrix:
            if str(word_id) in word_matrix[story]:
                word_freqs[word_id] += word_matrix[story][str(word_id)]
            else:
                continue
        word_freqs[top_words[word_id]] = word_freqs.pop(word_id)
    
    # importing modules for wordcloud
    from wordcloud import WordCloud, STOPWORDS
    import re
    import matplotlib.pyplot as plt
    
    # adding query to stopwords for wordcloud, so it doesn't dominate the cloud
    stopwords = set(STOPWORDS)
    pattern = re.compile('[\W_]+')
    query_split = query.lower().split()
    query_stops = {pattern.sub('', word) for word in query_split}
    stopwords.update(query_stops)
    if custom_stopwords:
        stopwords.update(custom_stopwords)
    for stopword in stopwords:
        if stopword in word_freqs:
            del word_freqs[stopword]
    wc_fig = plt.figure()
    word_cloud = WordCloud(background_color="white", width=3000, height=2000,
                           stopwords=stopwords, max_words=75, prefer_horizontal=1.0)
    word_cloud.fit_words(word_freqs)
    plt.imshow(word_cloud)
    plt.axis("off")
    plt.show()
    if save_img:
        if filename:
            word_cloud.to_file(filename)
        else:
            word_cloud.to_file("wordcloud.png")
    return wc_fig

# function to plot attention over time for one or more queries
def attention_plots(queries, sources, date_range, source_type="media", query_context=None,
                    api_key=None, query_labels=None, fig_size=(10,5), verbose=False):
    
    # initializing MediaCloud API, checking source format, cleaning date_range
    mc = init_mc_api(api_key=api_key)
    mc_source_type = check_source_type(sources, source_type)
    api_date_range = clean_api_date(mc, date_range) if date_range else None
    
    # formatting query for plots
    if type(queries) == str:
        queries = [queries]
    else:
        assert type(queries)==list, "Please pass either a list or string of queries into this function."
        
    # formatting media_ids for API query
    media_ids = list(sources.values())
    api_media_ids = " OR ".join(f"{mc_source_type}:{media_id}" for media_id in media_ids)
    
    # looping over each query and adding attention vs time plot to figure
    import matplotlib.pyplot as plt
    figure = plt.figure(figsize=fig_size)
    if query_labels:
        labels = query_labels
    else:
        labels = [f"Query {n}" for n in range(len(queries))]
    
    i = 0
    for query in queries:
        if verbose:
            print(f"Query {i}: {query}")
        
        # defining overall context and specific queries
        if query_context:
            total_query = f"({query_context}) and ({api_media_ids})"
        else:
            total_query = f"({api_media_ids})"
        relevant_query = f"({query}) and {total_query}"
        
        # making API calls for relevant and total story counts (by day)
        relevant_results = mc.storyCount(relevant_query, api_date_range, split=True, split_period='day')['counts']
        if not relevant_results:
            print(f"0 results for {query} between {start_date.strftime('%m/%d/%Y')} and {end_date.strftime('%m/%d/%Y')}.")
            continue
        total_results = mc.storyCount(total_query, api_date_range, split=True, split_period='day')['counts']
        
        # creating dataframes for query and total results
        import pandas as pd
        relevant_df = pd.DataFrame(relevant_results)
        relevant_df["date"] = pd.to_datetime(relevant_df["date"])
        total_df = pd.DataFrame(total_results)
        total_df["date"] = pd.to_datetime(total_df["date"])
        
        # joining dataframes on date, filling missing dates from query with zeros
        join_df = total_df.merge(relevant_df, how="outer", on="date", suffixes=("_total", "_relevant"))
        join_df["count_relevant"] = join_df["count_relevant"].fillna(0).astype(int)
        join_df = join_df[["date", "count_total", "count_relevant"]]
        
        # calculating attention
        join_df["attention"] = (join_df["count_relevant"] / join_df["count_total"]) * 100
        plt.plot(join_df["date"], join_df["attention"], '-', label=labels[i])
        
        i += 1
    
    # setting parameters for plot
    plt.suptitle("Attention over time")
    plt.legend(loc=0)
    plt.xlabel("Date")
    plt.ylabel("Attention in sources (%)")
    plt.xticks(rotation=60)
    plt.show()
    return figure
