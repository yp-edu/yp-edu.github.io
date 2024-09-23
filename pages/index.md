---
layout: home
permalink: /
---

## Welcome!

I am a French PhD student in AI with a focus on AI Safety. I've had diverse experience in research, education and industry. 
I'll outline this in a forthcoming story, "A Convoluted Study Path", giving insight into my career aspiration. 
More on me in the [About](/about) section.

You'll find here a collection of stories, projects and articles.

- Story: personal thought, claim, discussion or experience 
- Project: concrete application (mostly around AI)
- Article: paper explanation or discussion (definitely about AI)

## Pinned [stories](/stories/)  [::rss::](/stories/feed.xml)

{% assign pinned_stories = "/stories/my-approach-to-ai-safety, /stories/creating-a-startup, /stories/my-phd" | split: ", " %}
{% assign collection = site.stories | sort: "publishedOn" | reverse %}
{% for item in collection %}
{% if pinned_stories contains item.url %}

### [{{ item.title }}]({{ item.url }})
<div class="thumbnail">
  {% if item.image %}
    <a href="{{ item.url }}"><img src="{{ item.image }}" alt="{{ item.title }}" class="thumbnail" /></a>
  {% endif %}
  {% if item.tldr %}
  <p class="tldr">
    <small class="date">{{ item.publishedOn }} </small><small>| {{ item.readingTime }} min read</small><small> | {{ item.tldr }}</small>
  </p>
  {% endif %}
</div>
{% endif %}
{% endfor %}

## Pinned [projects](/projects/) [::rss::](/projects/feed.xml)

{% assign pinned_projects = "/projects/training-gpt2-on-stockfish-games" | split: ", " %}
{% assign collection = site.projects | sort: "publishedOn" | reverse %}
{% for item in collection %}
{% if pinned_projects contains item.url %}

### [{{ item.title }}]({{ item.url }})
<div class="thumbnail">
  {% if item.image %}
    <a href="{{ item.url }}"><img src="{{ item.image }}" alt="{{ item.title }}" class="thumbnail" /></a>
  {% endif %}
  {% if item.tldr %}
  <p class="tldr">
    <small class="date">{{ item.publishedOn }} </small><small>| {{ item.readingTime }} min read</small><small> | {{ item.tldr }}</small>
  </p>
  {% endif %}
</div>
{% endif %}
{% endfor %}

## Pinned [articles](/articles/) [::rss::](/articles/feed.xml)

{% assign pinned_articles = "/articles/layer-wise-relevance-propagation" | split: ", " %}
{% assign collection = site.articles | sort: "publishedOn" | reverse %}
{% for item in collection %}
{% if pinned_articles contains item.url %}

### [{{ item.title }}]({{ item.url }})
<div class="thumbnail">
  {% if item.image %}
    <a href="{{ item.url }}"><img src="{{ item.image }}" alt="{{ item.title }}" class="thumbnail" /></a>
  {% endif %}
  {% if item.tldr %}
  <p class="tldr">
    <small class="date">{{ item.publishedOn }} </small><small>| {{ item.readingTime }} min read</small><small> | {{ item.tldr }}</small>
  </p>
  {% endif %}
</div>
{% endif %}
{% endfor %}
