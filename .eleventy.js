// Filters
const dateFilter = require('./src/filters/date-filter.js');
const w3DateFilter = require('./src/filters/w3-date-filter.js');
const markdownIt = require("markdown-it");
const options = {};

const mdfigcaption = require('markdown-it-image-figures');
const figoptions = {
    figcaption: true
};

const mdLib = markdownIt(options).use(mdfigcaption, figoptions);

module.exports = function(config) {
   config.addPassthroughCopy("./src/style.css");
   config.addPassthroughCopy('./src/images/');
   config.addPassthroughCopy('./src/scripts/')

   // Add filters
   config.addFilter('dateFilter', dateFilter);
   config.addFilter('w3DateFilter', w3DateFilter);

   config.addCollection('blog', collection => {
      return [...collection.getFilteredByGlob('./src/blog/*.md')].reverse();
   });

   // Tell 11ty to use the .eleventyignore and ignore our .gitignore file
   config.setUseGitIgnore(false);

   config.setLibrary("md", mdLib);

     // Set custom directories for input, output, includes, and data  
   return {
      markdownTemplateEngine: 'njk',
      dataTemplateEngine: 'njk',
      htmlTemplateEngine: 'njk',

      passthroughFileCopy: true,
      dir: {
         input: "src",
         includes: "_includes",
         data: "_data",
         output: "dist"
      }  
   };
};