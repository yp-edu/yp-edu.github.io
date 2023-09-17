
require "jekyll-commonmark"
require 'kramdown'

FA_CALLOUTS = {
  "tip" => "fa-solid fa-fire-flame-curved",
  "hint" => "fa-solid fa-fire-flame-curved",
  "important" => "fa-solid fa-fire-flame-curved",
  "abstract" => "fa-solid fa-clipboard-list",
  "summary" => "fa-solid fa-clipboard-list",
  "tldr" => "fa-solid fa-clipboard-list",
  "info" => "fa-solid fa-circle-info",
  "bug" => "fa-solid fa-bug",
  "note" => "fa-solid fa-pen",
  "rss" => "fa-solid fa-rss",
  "linkedin" => "fa-brands fa-linkedin",
}
DEBUG = false

class Jekyll::Converters::Markdown
  class CustomMarkProcessor < CommonMark
    def convert(content)
      html = Kramdown::Document.new(content).to_html
      if DEBUG
        Jekyll.logger.info "CustomMarkProcessor:", "content class: #{content.class}"
        Jekyll.logger.info "CustomMarkProcessor:", "html class: #{html.class}"
      end
      if DEBUG
        Jekyll.logger.info "CustomMarkProcessor:", "html: #{html}"
      end
      regexp = /<blockquote>\s*.*<p>\[\!([a-z]{1,})\]\s?(.*)<\/p>\s*.*<p>(.*)<\/p>\s*.*<\/blockquote>/
      matches = html.match(regexp)
      parsed_html = html.gsub(
        regexp, 
        '<blockquote class="callout \1"> <div class="callout-title"> '\
        '<i class="fa-\1" href="#"></i> '\
        '<em>\2</em></div> <p>\3</p> </blockquote>'
        )
      parsed_html = parsed_html.gsub(/::(.*)::/, '<i class="fa-\1"></i>')
      FA_CALLOUTS.each do |key, value|
        parsed_html.gsub!(/<i class="fa-#{key}"/, "<i class=\"#{value}\"")
      end
      if DEBUG
        Jekyll.logger.info "CustomMarkProcessor:", "matches: #{matches}"
        Jekyll.logger.info "CustomMarkProcessor:", "parsed_html: #{parsed_html}"
      end
      parsed_html = parsed_html.gsub(/<h([1-6])\sid="([a-z]|\-)*">/, '<h\1>')
      parsed_html = parsed_html.gsub(/<h2>(\s*.*)<\/h2>/, '<h2># \1</h2>')
      parsed_html
    end
  end
end